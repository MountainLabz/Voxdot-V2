use crate::morton;
use crate::octree::OctreeManager;
use crate::voxdot_terrain::FloatingClump;
use crate::voxdot_terrain::FloatingVoxel;
use crate::world_classes::BakedRegistry;
use crate::world_classes::ChunkState;
use crate::world_classes::GpuUploadData;
use crate::world_classes::TerrainManagerData;
use crate::world_classes::VoxelDictionary;
use godot::classes::BoxMesh;
use godot::classes::BoxShape3D;
use godot::classes::IRigidBody3D;
use godot::classes::MultiMesh;
use godot::classes::MultiMeshInstance3D;
use godot::classes::ProjectSettings;
use godot::classes::RigidBody3D;
use godot::classes::image::Format;
use godot::classes::multi_mesh::TransformFormat;
use godot::classes::rendering_device::MemoryType;
use godot::classes::rendering_server::PrimitiveType;
use godot::classes::{
    ArrayMesh, CollisionShape3D, ConcavePolygonShape3D, IMeshInstance3D, IStaticBody3D, Mesh,
    MeshInstance3D, Node3D, PhysicsServer3D, RenderingDevice, Shader, ShaderMaterial, StaticBody3D,
    TriangleMesh,
};
use godot::classes::{Image, ImageTexture3D};
use godot::obj::NewAlloc;
use godot::prelude::*;
use godot::prelude::*;
use hashbrown::HashMap;
use mint::Point3;
use rustc_hash::FxHasher;
use std::fs::File;
use std::sync::OnceLock;
//use godot::engine::mesh::PrimitiveType;???
//use godot::classes::PrimitiveMesh;
use crate::build;
use godot::classes::FastNoiseLite;
use godot::classes::RenderingServer;
use godot::classes::fast_noise_lite::{FractalType, NoiseType};
use godot::classes::mesh::ArrayType;
//use godot::classes::mesh::PrimitiveType;
use std::hash::BuildHasherDefault;
use std::ops::Deref;
// Add 'rand' to your Cargo.toml
use fastnoise2;
use std::io::Read;
use std::time::Instant;
use wide::u64x4;

use std::cell::RefCell;

const VOXEL_SHADER_CODE: &str = include_str!("shaders/rigidbody.gdshader");

thread_local! {
    static DEBRIS_MATERIAL: RefCell<Option<Gd<ShaderMaterial>>> = RefCell::new(None);
}

fn get_shared_debris_material() -> Gd<ShaderMaterial> {
    DEBRIS_MATERIAL.with(|storage| {
        let mut storage = storage.borrow_mut();

        if let Some(mat) = &*storage {
            return mat.clone();
        }

        // 3. Create Shader from the embedded string
        let mut shader = Shader::new_gd();
        shader.set_code(VOXEL_SHADER_CODE);

        // 4. Create the Material
        let mut mat = ShaderMaterial::new_gd();
        mat.set_shader(&shader);

        *storage = Some(mat.clone());
        mat
    })
}

pub enum ClumpBuffer {
    Small {
        visuals: Box<[u32; 32768]>, // 32^3
        occupancy: Box<[u64; 512]>, // 32^3 / 64
        gpu_buffer: Box<[u32; 32768]>,
    },
    Medium {
        visuals: Box<[u32; 262144]>, // 64^3
        occupancy: Box<[u64; 4096]>, // 64^3 / 64
        gpu_buffer: Box<[u32; 262144]>,
    },
    Large {
        visuals: Box<[u32; 2097152]>, // 128^3
        occupancy: Box<[u64; 32768]>, // 128^3 / 64
        gpu_buffer: Box<[u32; 2097152]>,
    },
    Huge {
        visuals: Box<[u32; 16777216]>, // 256^3
        occupancy: Box<[u64; 262144]>, // 256^3 / 64
        gpu_buffer: Box<[u32; 16777216]>,
    },
}

impl ClumpBuffer {
    /// Returns the side dimension for the current variant
    pub fn side(&self) -> i32 {
        match self {
            ClumpBuffer::Small { .. } => 32,
            ClumpBuffer::Medium { .. } => 64,
            ClumpBuffer::Large { .. } => 128,
            ClumpBuffer::Huge { .. } => 256,
        }
    }

    pub fn gpu_slice(&self) -> &[u32] {
        match self {
            ClumpBuffer::Small { gpu_buffer, .. } => gpu_buffer.as_slice(),
            ClumpBuffer::Medium { gpu_buffer, .. } => gpu_buffer.as_slice(),
            ClumpBuffer::Large { gpu_buffer, .. } => gpu_buffer.as_slice(),
            ClumpBuffer::Huge { gpu_buffer, .. } => gpu_buffer.as_slice(),
        }
    }

    pub fn occupancy_slice(&self) -> &[u64] {
        match self {
            ClumpBuffer::Small { occupancy, .. } => occupancy.as_slice(),
            ClumpBuffer::Medium { occupancy, .. } => occupancy.as_slice(),
            ClumpBuffer::Large { occupancy, .. } => occupancy.as_slice(),
            ClumpBuffer::Huge { occupancy, .. } => occupancy.as_slice(),
        }
    }

    /// Flat index: z * (side^2) + y * side + x
    pub fn get_index(&self, x: i32, y: i32, z: i32) -> usize {
        let s = self.side();
        ((z * s * s) + (y * s) + x) as usize
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
pub struct VoxelRigidBody {
    #[base]
    base: Base<RigidBody3D>,

    pub data: Option<ClumpBuffer>,

    #[export]
    pub mesh_resource: Option<Gd<Mesh>>, // The 1x1x1 unit box

    #[export]
    pub enable_occlusion: bool, // NEW: Toggle occlusion culling separately

    pub multimesh_instance: Option<Gd<MultiMeshInstance3D>>,
    pub voxel_material: Option<Gd<ShaderMaterial>>,
}

#[godot_api]
impl IRigidBody3D for VoxelRigidBody {
    fn init(base: Base<RigidBody3D>) -> Self {
        // 1. Create the 0.1x0.1x0.1 box mesh
        let mut box_mesh = BoxMesh::new_gd();
        box_mesh.set_size(Vector3::new(0.1, 0.1, 0.1));

        base.to_init_gd().set_can_sleep(false);

        // 2. Return the struct with default values
        Self {
            base,
            data: None,
            mesh_resource: Some(box_mesh.upcast()),
            enable_occlusion: true,
            multimesh_instance: None,
            voxel_material: Some(get_shared_debris_material()),
        }
    }
}

#[godot_api]
impl VoxelRigidBody {
    #[inline(always)]
    fn is_voxel_solid(occupancy: &[u64], x: i32, y: i32, z: i32, side: i32) -> bool {
        if x < 0 || x >= side || y < 0 || y >= side || z < 0 || z >= side {
            return false;
        }
        let idx = (z * side * side + y * side + x) as usize;
        (occupancy[idx / 64] >> (idx % 64)) & 1 == 1
    }

    #[inline(always)]
    fn calculate_clump_normal(occupancy: &[u64], x: i32, y: i32, z: i32, side: i32) -> Vector3 {
        let mut nx = 0.0;
        let mut ny = 0.0;
        let mut nz = 0.0;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    if Self::is_voxel_solid(occupancy, x + dx, y + dy, z + dz, side) {
                        nx -= dx as f32;
                        ny -= dy as f32;
                        nz -= dz as f32;
                    }
                }
            }
        }

        let length_sq = nx * nx + ny * ny + nz * nz;
        if length_sq == 0.0 {
            return Vector3::new(0.0, 1.0, 0.0);
        }

        if length_sq > 0.01 {
            let inv_len = 1.0 / length_sq.sqrt();
            Vector3::new(nx * inv_len, ny * inv_len, nz * inv_len)
        } else {
            Vector3::UP
        }
    }

    fn allocate_data(size_tier: i32) -> ClumpBuffer {
        // Helper to avoid stack overflows
        fn safe_box_u32<const N: usize>() -> Box<[u32; N]> {
            let v = vec![0u32; N];
            let boxed = v.into_boxed_slice();
            boxed
                .try_into()
                .unwrap_or_else(|_| panic!("Failed to allocate {} bytes on heap", N * 4))
        }

        fn safe_box_u64<const N: usize>() -> Box<[u64; N]> {
            let v = vec![0u64; N];
            let boxed = v.into_boxed_slice();
            boxed
                .try_into()
                .unwrap_or_else(|_| panic!("Failed to allocate {} bytes on heap", N * 8))
        }

        match size_tier {
            64 => ClumpBuffer::Medium {
                visuals: safe_box_u32::<262144>(),
                occupancy: safe_box_u64::<4096>(),
                gpu_buffer: safe_box_u32::<262144>(),
            },
            128 => ClumpBuffer::Large {
                visuals: safe_box_u32::<2097152>(),
                occupancy: safe_box_u64::<32768>(),
                gpu_buffer: safe_box_u32::<2097152>(),
            },
            256 => ClumpBuffer::Huge {
                visuals: safe_box_u32::<16777216>(),
                occupancy: safe_box_u64::<262144>(),
                gpu_buffer: safe_box_u32::<16777216>(),
            },
            _ => ClumpBuffer::Small {
                visuals: safe_box_u32::<32768>(),
                occupancy: safe_box_u64::<512>(),
                gpu_buffer: safe_box_u32::<32768>(),
            },
        }
    }

    pub fn init_clump(&mut self, size_tier: i32, clump: FloatingClump) {
        // Automatically create the 0.1x0.1x0.1 box mesh on initialization
        // let mut box_mesh = BoxMesh::new_gd();
        // box_mesh.set_size(Vector3::new(0.1, 0.1, 0.1));

        // self.mesh_resource = Some(box_mesh.upcast());

        self.voxel_material = Some(get_shared_debris_material());
        // 1. Allocate based on tier
        self.data = Some(Self::allocate_data(size_tier));

        // 2. Populate buffers and sync MultiMesh
        let (min, max) = self.fill_buffers(clump.voxels);
        self.bake_gpu_buffer();
        self.rebuild_multimesh();

        self.apply_tight_collision(
            Vector3::new(min.x as f32 * 0.1, min.y as f32 * 0.1, min.z as f32 * 0.1),
            Vector3::new(max.x as f32 * 0.1, max.y as f32 * 0.1, max.z as f32 * 0.1),
        );

        // 3. Set Physics AABB
        //self.update_aabb_collision(size_tier as f32);

        // 4. Set Initial Position
        self.base_mut()
            .set_global_position(clump.world_origin * Vector3::new(0.1, 0.1, 0.1));
    }

    pub fn setup_clump(&mut self, size_tier: i32, clump: FloatingClump) {
        // 1. Allocate the fixed-size buffers using the helper
        self.data = Some(Self::allocate_data(size_tier));

        // 2. Fill the CPU buffers from the sparse data
        self.fill_buffers(clump.voxels);

        // 3. Process the GPU buffer (Calculates smooth normals/packing)
        self.bake_gpu_buffer();

        // 4. Update MultiMesh (Rendering)
        self.rebuild_multimesh();

        // 5. Position in world
        self.base_mut().set_global_position(clump.world_origin);
    }

    fn fill_buffers(&mut self, voxels: Vec<FloatingVoxel>) -> (Vector3i, Vector3i) {
        let mut min = Vector3i::new(i32::MAX, i32::MAX, i32::MAX);
        let mut max = Vector3i::new(i32::MIN, i32::MIN, i32::MIN);
        if let Some(ref mut data) = self.data {
            let side = data.side();

            for v in voxels {
                let p = v.local_pos;
                // Bounds check
                if p.x >= 0 && p.x < side && p.y >= 0 && p.y < side && p.z >= 0 && p.z < side {
                    let idx = data.get_index(p.x, p.y, p.z);

                    min.x = min.x.min(p.x);
                    min.y = min.y.min(p.y);
                    min.z = min.z.min(p.z);
                    max.x = max.x.max(p.x);
                    max.y = max.y.max(p.y);
                    max.z = max.z.max(p.z);

                    match data {
                        ClumpBuffer::Small {
                            visuals, occupancy, ..
                        } => {
                            visuals[idx] = v.color_data;
                            occupancy[idx / 64] |= 1 << (idx % 64);
                        }
                        ClumpBuffer::Medium {
                            visuals, occupancy, ..
                        } => {
                            visuals[idx] = v.color_data;
                            occupancy[idx / 64] |= 1 << (idx % 64);
                        }
                        ClumpBuffer::Large {
                            visuals, occupancy, ..
                        } => {
                            visuals[idx] = v.color_data;
                            occupancy[idx / 64] |= 1 << (idx % 64);
                        }
                        ClumpBuffer::Huge {
                            visuals, occupancy, ..
                        } => {
                            visuals[idx] = v.color_data;
                            occupancy[idx / 64] |= 1 << (idx % 64);
                        }
                    }
                }
            }
        }
        (min, max)
    }

    pub fn rebuild_multimesh(&mut self) {
        let data = match &self.data {
            Some(d) => d,
            None => return,
        };
        let side = data.side();
        let occupancy = data.occupancy_slice();

        // Pass 1: Count exposed voxels for buffer sizing
        let mut visible_count = 0;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    if !Self::is_voxel_solid(occupancy, x, y, z, side) {
                        continue;
                    }
                    if self.is_exposed(occupancy, x, y, z, side) {
                        visible_count += 1;
                    }
                }
            }
        }

        if visible_count == 0 {
            return;
        }

        let mut mm = MultiMesh::new_gd();
        mm.set_transform_format(TransformFormat::TRANSFORM_3D);
        mm.set_use_custom_data(true);
        mm.set_instance_count(visible_count as i32);

        let mut mesh = BoxMesh::new_gd();
        mesh.set_size(Vector3::new(0.1, 0.1, 0.1));
        mm.set_mesh(mesh.to_godot());

        // Pass 2: Fill MultiMesh data
        let mut curr = 0;
        let gpu_slice = data.gpu_slice(); // Add a helper to ClumpBuffer to return &gpu_buffer
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    if !Self::is_voxel_solid(occupancy, x, y, z, side) {
                        continue;
                    }
                    if self.is_exposed(occupancy, x, y, z, side) {
                        let transform = Transform3D::new(
                            Basis::IDENTITY,
                            Vector3::new(x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1),
                        );
                        mm.set_instance_transform(curr, transform);

                        let p = gpu_slice[data.get_index(x, y, z)];
                        let custom_data = Color::from_rgba(
                            ((p >> 24) & 0xFF) as f32 / 255.0,
                            ((p >> 16) & 0xFF) as f32 / 255.0,
                            ((p >> 8) & 0xFF) as f32 / 255.0,
                            (p & 0xFF) as f32 / 255.0,
                        );
                        mm.set_instance_custom_data(curr, custom_data);
                        curr += 1;
                    }
                }
            }
        }

        // Pass 3: Material application
        if let Some(mut old_mmi) = self.multimesh_instance.take() {
            old_mmi.queue_free();
        }
        let mut mmi = MultiMeshInstance3D::new_alloc();
        mmi.set_multimesh(&mm);

        // IMPORTANT: Load your shader material here
        // Replace "res://path/to/your_material.tres" with the actual path
        if let Some(mat) = self.voxel_material.as_ref() {
            mmi.set_material_override(mat.to_godot());
        }

        self.base_mut().add_child(mmi.clone().to_godot());
        self.multimesh_instance = Some(mmi);
    }

    // Helper for visibility culling
    fn is_exposed(&self, occ: &[u64], x: i32, y: i32, z: i32, side: i32) -> bool {
        !Self::is_voxel_solid(occ, x + 1, y, z, side)
            || !Self::is_voxel_solid(occ, x - 1, y, z, side)
            || !Self::is_voxel_solid(occ, x, y + 1, z, side)
            || !Self::is_voxel_solid(occ, x, y - 1, z, side)
            || !Self::is_voxel_solid(occ, x, y, z + 1, side)
            || !Self::is_voxel_solid(occ, x, y, z - 1, side)
    }

    fn apply_tight_collision(&mut self, min: Vector3, max: Vector3) {
        // Remove old shapes
        for child in self.base().get_children().iter_shared() {
            // Use .ok() to convert the Result to an Option for if let
            if let Some(mut shape) = child.try_cast::<CollisionShape3D>().ok() {
                shape.queue_free();
            }
        }

        let size = max - min;
        let center = min + (size * 0.5);

        let mut shape_node = CollisionShape3D::new_alloc();
        let mut box_res = BoxShape3D::new_gd();
        box_res.set_size(size);

        shape_node.set_shape(box_res.to_godot());
        shape_node.set_position(center);

        self.base_mut().add_child(shape_node.to_godot());

        // Optional: Re-balance the physics so it doesn't spin around a ghost corner
        self.base_mut()
            .set_center_of_mass_mode(godot::classes::rigid_body_3d::CenterOfMassMode::CUSTOM);
        self.base_mut().set_center_of_mass(center);
    }

    pub fn bake_gpu_buffer(&mut self) {
        let data_ref = match &mut self.data {
            Some(d) => d,
            None => return,
        };

        let side = data_ref.side();
        let side_sq = side * side; // Pre-calculate for the index math

        // 1. Extract all 3 slices (visuals, occupancy, gpu_buffer) in ONE match.
        // This allows Rust to "split" the borrow of the Enum variant into its fields.
        let (visuals, occupancy, gpu_buffer) = match data_ref {
            ClumpBuffer::Small {
                visuals,
                occupancy,
                gpu_buffer,
                ..
            } => (
                visuals.as_slice(),
                occupancy.as_slice(),
                gpu_buffer.as_mut_slice(),
            ),
            ClumpBuffer::Medium {
                visuals,
                occupancy,
                gpu_buffer,
                ..
            } => (
                visuals.as_slice(),
                occupancy.as_slice(),
                gpu_buffer.as_mut_slice(),
            ),
            ClumpBuffer::Large {
                visuals,
                occupancy,
                gpu_buffer,
                ..
            } => (
                visuals.as_slice(),
                occupancy.as_slice(),
                gpu_buffer.as_mut_slice(),
            ),
            ClumpBuffer::Huge {
                visuals,
                occupancy,
                gpu_buffer,
                ..
            } => (
                visuals.as_slice(),
                occupancy.as_slice(),
                gpu_buffer.as_mut_slice(),
            ),
        };

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    // 2. Calculate the index manually to avoid calling methods on data_ref.
                    // This assumes standard (Z * side^2 + Y * side + X) layout.
                    let idx = (z * side_sq + y * side + x) as usize;

                    // Skip empty voxels
                    if !Self::is_voxel_solid(occupancy, x, y, z, side) {
                        continue;
                    }

                    // Exposure logic (6-way check)
                    let exposed = !Self::is_voxel_solid(occupancy, x + 1, y, z, side)
                        || !Self::is_voxel_solid(occupancy, x - 1, y, z, side)
                        || !Self::is_voxel_solid(occupancy, x, y + 1, z, side)
                        || !Self::is_voxel_solid(occupancy, x, y - 1, z, side)
                        || !Self::is_voxel_solid(occupancy, x, y, z + 1, side)
                        || !Self::is_voxel_solid(occupancy, x, y, z - 1, side);

                    if !exposed {
                        continue;
                    }

                    // 3. Extract base data from CPU visuals
                    let cpu_packed = visuals[idx];
                    let color_15b = cpu_packed & 0x7FFF;
                    let extras = (cpu_packed >> 24) & 0xF;

                    // 4. Calculate Normals
                    let normal = Self::calculate_clump_normal(occupancy, x, y, z, side);
                    let nx_4b = ((normal.x + 1.0) * 0.5 * 15.0).round() as u32;
                    let ny_4b = ((normal.y + 1.0) * 0.5 * 15.0).round() as u32;
                    let nz_4b = ((normal.z + 1.0) * 0.5 * 15.0).round() as u32;

                    // 5. Pack for GPU: [4b Extras][4b NZ][4b NY][4b NX][15b Color]
                    let final_gpu_val = color_15b
                        | (nx_4b << 15)
                        | (ny_4b << 19)
                        | (nz_4b << 23)
                        | ((extras & 0xF) << 27);

                    gpu_buffer[idx] = final_gpu_val;
                }
            }
        }
    }

    // fn rebuild_multimesh(&mut self) {
    //     let data_ref = match &self.data {
    //         Some(d) => d,
    //         None => return,
    //     };

    //     // 1. Count active voxels
    //     let active_count: u32 = data_ref
    //         .occupancy_slice()
    //         .iter()
    //         .map(|word| word.count_ones())
    //         .sum();

    //     if active_count == 0 {
    //         return;
    //     }

    //     // 2. Setup MultiMesh
    //     let mut mm = MultiMesh::new_gd();
    //     // Use Transform3D for the mesh storage format
    //     mm.set_transform_format(godot::classes::multi_mesh::TransformFormat::TRANSFORM_3D);
    //     mm.set_use_custom_data(true);

    //     let mesh = self
    //         .mesh_resource
    //         .clone()
    //         .expect("Missing Voxel Mesh Resource");
    //     mm.set_mesh(&mesh);
    //     mm.set_instance_count(active_count as i32);

    //     // 3. Populate instances
    //     let side = data_ref.side();
    //     let mut current_instance = 0;
    //     let voxel_scale: f32 = 0.1; // CRITICAL: Match your box size

    //     for z in 0..side {
    //         for y in 0..side {
    //             for x in 0..side {
    //                 let idx = data_ref.get_index(x, y, z);

    //                 // Helper to check occupancy
    //                 let is_active = match data_ref {
    //                     ClumpBuffer::Small { occupancy, .. } => {
    //                         (occupancy[idx / 64] >> (idx % 64)) & 1 == 1
    //                     }
    //                     ClumpBuffer::Medium { occupancy, .. } => {
    //                         (occupancy[idx / 64] >> (idx % 64)) & 1 == 1
    //                     }
    //                     ClumpBuffer::Large { occupancy, .. } => {
    //                         (occupancy[idx / 64] >> (idx % 64)) & 1 == 1
    //                     }
    //                     ClumpBuffer::Huge { occupancy, .. } => {
    //                         (occupancy[idx / 64] >> (idx % 64)) & 1 == 1
    //                     }
    //                 };

    //                 if is_active {
    //                     let visual_u32 = match data_ref {
    //                         ClumpBuffer::Small { visuals, .. } => visuals[idx],
    //                         ClumpBuffer::Medium { visuals, .. } => visuals[idx],
    //                         ClumpBuffer::Large { visuals, .. } => visuals[idx],
    //                         ClumpBuffer::Huge { visuals, .. } => visuals[idx],
    //                     };

    //                     // SCALE THE POSITION: multiply loop indices by voxel_scale
    //                     // Otherwise voxels are 1.0 units apart but only 0.1 units big
    //                     let pos = Vector3::new(
    //                         x as f32 * voxel_scale,
    //                         y as f32 * voxel_scale,
    //                         z as f32 * voxel_scale,
    //                     );

    //                     let transform = Transform3D::IDENTITY.translated(pos);
    //                     mm.set_instance_transform(current_instance, transform);

    //                     // Map the u32 to Color channels for the shader
    //                     let packed_color = Color::from_rgba8(
    //                         ((visual_u32 >> 24) & 0xFF) as u8,
    //                         ((visual_u32 >> 16) & 0xFF) as u8,
    //                         ((visual_u32 >> 8) & 0xFF) as u8,
    //                         (visual_u32 & 0xFF) as u8,
    //                     );
    //                     mm.set_instance_custom_data(current_instance, packed_color);

    //                     current_instance += 1;
    //                 }
    //             }
    //         }
    //     }

    //     // 4. Create and attach the Instance node
    //     // Cleanup old instance if it exists (prevents ghost meshes when updating)
    //     if let Some(mut old_mmi) = self.multimesh_instance.take() {
    //         old_mmi.queue_free();
    //     }

    //     let mut mmi = MultiMeshInstance3D::new_alloc();
    //     mmi.set_multimesh(&mm);

    //     mmi.set_material_override(get_shared_debris_material().to_godot());
    //     // IMPORTANT: If your boxes use a custom shader to unpack the colors,
    //     // you MUST ensure the MultiMeshInstance has that material assigned.
    //     // If the material is on the Mesh itself, this is fine.

    //     self.base_mut().add_child(mmi.clone().to_godot());
    //     self.multimesh_instance = Some(mmi);
    // }

    // fn update_aabb_collision(&mut self, size: f32) {
    //     let mut shape = CollisionShape3D::new_alloc();
    //     let mut box_res = BoxShape3D::new_gd();
    //     box_res.set_size(Vector3::new(size * 0.1, size * 0.1, size * 0.1));
    //     shape.set_shape(box_res.to_godot());
    //     // Offset by half-size if the clump origin is the corner
    //     shape.set_position(
    //         Vector3::new(size * 0.5, size * 0.5, size * 0.5) * Vector3::new(0.1, 0.1, 0.1),
    //     );
    //     self.base_mut().add_child(shape.to_godot());
    // }
}
