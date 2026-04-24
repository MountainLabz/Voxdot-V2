use crate::morton;
use crate::octree::OctreeManager;
use crate::voxdot_rigidbody::VoxelRigidBody;
use crate::world_classes::BakedRegistry;
use crate::world_classes::ChunkState;
use crate::world_classes::GpuUploadData;
use crate::world_classes::LoadedWorld;
use crate::world_classes::TerrainManagerData;
use crate::world_classes::VoxelDictionary;
use crate::world_classes::WorkItem;
use godot::classes::ProjectSettings;
use godot::classes::image::Format;
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

///
///
///  Voxel Structs
///
///

type FxBuildHasher = BuildHasherDefault<FxHasher>;
type ChunkMap = HashMap<u128, ChunkData, FxBuildHasher>;

// Coordinate Offset to allow negative chunk coordinates in Morton space
// 2^31 = 2,147,483,648. This centers 0,0,0 in the middle of the unsigned range.
const COORD_OFFSET: i32 = 2_147_483_647_i32.wrapping_div(2) as i32; // ~1 billion buffer
const VOXEL_SHADER_CODE: &str = include_str!("shaders/chunk_sparse.gdshader");

#[derive(Clone)]
pub struct ChunkData {
    // occupancy data binary representation stuff.
    pub occupancy: Box<[u64; 4096]>,
    // DETAILED DATA STUFF: Used for the 3D Texture (Color/Material)
    // Flattened 64^3 array. Index = z * 4096 + y * 64 + x
    pub visuals: Box<[u32; 262_144]>,
    pub mesh_rid: Rid,
    pub instance_rid: Rid,
    pub material_rid: Rid,

    //legacy
    pub texture_rid: Rid,

    // New GPU Structs
    pub bitmask_rid: Rid,
    pub index_rid: Rid,
    pub visuals_rid: Rid,

    pub occluder_rid: Rid,
    pub occluder_instance_rid: Rid,

    pub collider: Option<Gd<CollisionShape3D>>,
    pub cached_faces: Option<PackedVector3Array>,

    // meta
    pub is_dirty: bool,
}

impl ChunkData {
    pub fn new() -> Self {
        Self {
            occupancy: Box::new([0u64; 4096]),
            visuals: Box::new([0u32; 262_144]), // 1MB allocation
            mesh_rid: Rid::Invalid,
            instance_rid: Rid::Invalid,
            material_rid: Rid::Invalid,
            texture_rid: Rid::Invalid,
            bitmask_rid: Rid::Invalid,
            index_rid: Rid::Invalid,
            visuals_rid: Rid::Invalid,
            occluder_rid: Rid::Invalid,
            occluder_instance_rid: Rid::Invalid,
            collider: None,
            cached_faces: None,
            is_dirty: false,
        }
    }
    pub fn calculate_density(&self) -> f32 {
        let mut total_ones = 0u64;
        for row in self.occupancy.iter() {
            total_ones += row.count_ones() as u64;
        }
        // A 64x64x64 chunk has 262,144 total possible voxels
        total_ones as f32 / 262144.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FloatingVoxel {
    pub local_pos: Vector3i, // Relative to the clump's origin
    pub color_data: u32,     // The raw 32-bit visual data from your visuals array
}

#[derive(Debug)]
pub struct FloatingClump {
    pub world_origin: Vector3, // Where to place the RigidBody3D in Godot
    pub voxels: Vec<FloatingVoxel>,
}

///
///
///  Voxel Nodes
///
///

#[derive(GodotClass)]
#[class(base=StaticBody3D)]
pub struct VoxelTerrain {
    base: Base<StaticBody3D>,
    // The core storage
    pub octree: OctreeManager,
    // Noise generator
    #[var]
    noise: Gd<FastNoiseLite>,
    #[export]
    voxel_size: f32,
    #[export]
    pub terrain_manager: Option<Gd<TerrainManagerData>>,
    #[export]
    material: Option<Gd<ShaderMaterial>>,
    #[export]
    pub enable_occlusion: bool,
    #[export]
    pub occlusion__density_threshold: f32,
    pub shader_rid: Rid, // NEW: shared shader RID for all chunks
    pub gpu_buffer: Box<[u32; 262_144]>,
    pub gpu_upload_data: GpuUploadData,

    pub world_state: LoadedWorld,
    #[export]
    pub lod_radii: Array<f32>, // 5 elements: [0.0, lod1_r, lod2_r, lod3_r, lod4_r]
    #[export]
    pub lod_scan_half_extent: i32, // how many LOD-4 nodes to scan per axis from camera
    #[export]
    pub chunk_size: f32,
}

#[godot_api]
impl IStaticBody3D for VoxelTerrain {
    fn init(base: Base<StaticBody3D>) -> Self {
        // Initialize noise
        let mut noise = FastNoiseLite::new_gd();
        noise.set_noise_type(NoiseType::PERLIN);
        noise.set_frequency(0.02);
        noise.set_fractal_type(FractalType::FBM);
        noise.set_fractal_octaves(3);

        let mut rs = RenderingServer::singleton();

        // create shared shader (do this once)
        let shader_rid = rs.shader_create();

        // VOXEL_SHADER_CODE should be a &str that contains your shader source
        rs.shader_set_code(shader_rid, VOXEL_SHADER_CODE);

        // store it in the terrain struct

        Self {
            base,
            octree: OctreeManager::new(), // New Octree Init
            noise,
            voxel_size: 0.1,
            terrain_manager: None,
            material: None,
            enable_occlusion: false,
            occlusion__density_threshold: 0.3,
            shader_rid,
            gpu_buffer: Box::new([0u32; 262_144]), // 1MB allocation
            gpu_upload_data: GpuUploadData::new(),
            world_state: LoadedWorld::new(),
            lod_scan_half_extent: 16,
            lod_radii: {
                let mut a = Array::new();
                a.push(0.0_f32); // depth 0 — leaf, no split radius needed
                a.push(64.0_f32); // depth 1 splits within  64 world-units
                a.push(128.0_f32); // depth 2 splits within 128 world-units
                a.push(256.0_f32); // depth 3 splits within 256 world-units
                a.push(512.0_f32); // depth 4 splits within 512 world-units
                a
            },
            chunk_size: 6.4,
        }
    }

    fn ready(&mut self) {
        // Example: Generate a 2x2x2 area around origin on startup
        // self.generate_region(0, 0, 0, 2);
        if self.material.is_none() {
            // let shader = load::<Shader>("res://voxel_shader.gdshader");
            // let mut mat = ShaderMaterial::new_gd();
            // mat.set_shader(shader);
            // self.material = Some(mat);
        }
    }

    // Cleanup RIDs when the terrain is destroyed to prevent VRAM leaks

    fn exit_tree(&mut self) {
        let mut rs = RenderingServer::singleton();

        // 1. Iterate through all nodes in the octree pool to free chunk resources
        for node in self.octree.node_pool.iter_mut() {
            if let Some(chunk) = &mut node.chunk {
                if chunk.instance_rid.is_valid() {
                    rs.free_rid(chunk.instance_rid);
                }
                if chunk.mesh_rid.is_valid() {
                    rs.free_rid(chunk.mesh_rid);
                }
                if chunk.texture_rid.is_valid() {
                    rs.free_rid(chunk.texture_rid);
                }
                if chunk.material_rid.is_valid() {
                    rs.free_rid(chunk.material_rid);
                }
                // Add this inside BOTH exit_tree() and remove_chunk() where you free other RIDs:
                if chunk.occluder_instance_rid.is_valid() {
                    rs.free_rid(chunk.occluder_instance_rid);
                }
                if chunk.occluder_rid.is_valid() {
                    rs.free_rid(chunk.occluder_rid);
                }

                if let Some(node) = &mut chunk.collider {
                    node.queue_free();
                }
            }
        }

        // 2. Free the shared shader
        if self.shader_rid.is_valid() {
            rs.free_rid(self.shader_rid);
            self.shader_rid = Rid::Invalid;
        }
    }
}

#[godot_api]
impl VoxelTerrain {
    // -------------------------------------------------------------------------
    //  1. Chunk Lifecycle Management
    // -------------------------------------------------------------------------

    #[func]
    pub fn create_chunk(&mut self, cx: i32, cy: i32, cz: i32, depth: u8, terrain: bool) {
        self.octree.ensure_chunk_mut(cx, cy, cz, depth);
    }

    #[func]
    pub fn remove_chunk(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
        let mut rs = RenderingServer::singleton();

        if let Some(idx) = self.octree.get_or_create_node(cx, cy, cz, depth, false) {
            let mut node = &mut self.octree.node_pool[idx as usize];
            if let Some(chunk) = &mut node.chunk {
                if chunk.instance_rid.is_valid() {
                    rs.free_rid(chunk.instance_rid);
                }
                if chunk.mesh_rid.is_valid() {
                    rs.free_rid(chunk.mesh_rid);
                }
                if chunk.texture_rid.is_valid() {
                    rs.free_rid(chunk.texture_rid);
                }
                if chunk.material_rid.is_valid() {
                    rs.free_rid(chunk.material_rid);
                }

                if let Some(node) = &mut chunk.collider {
                    node.queue_free();
                }
            }
        }
    }

    #[func]
    pub fn prepare_chunk_data(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) {
        if let Some(mut tm) = self.terrain_manager.clone() {
            tm.bind_mut().generate_biome_data(cx, cy, cz, depth);
            tm.bind_mut().generate_density_data(cx, cy, cz, depth);
        }
    }

    #[func]
    pub fn lodify_node(&mut self, nx: i32, ny: i32, nz: i32, depth: u8) {
        // 1. FIRST: Ensure the LOD node exists so we can sample into it
        let idx = if let Some(idx) = self.octree.get_or_create_node(nx, ny, nz, depth, true) {
            idx
        } else {
            return;
        };

        // 2. SECOND: Sample the visuals from the children while they still exist!
        let lod_visuals = self.octree.sample_visuals(idx);

        // 3. THIRD: Now it is safe to remove the children meshes/data
        let range = 1 << depth;
        for x in 0..range {
            for y in 0..range {
                for z in 0..range {
                    self.remove_chunk(nx + x, ny + y, nz + z, depth - 1);
                }
            }
        }

        // 4. FOURTH: Finalize the LOD node data
        let node = &mut self.octree.node_pool[idx as usize];
        if node.chunk.is_none() {
            node.chunk = Some(ChunkData::new());
        }

        if let Some(chunk) = &mut node.chunk {
            chunk.visuals = lod_visuals.0;

            // IMPORTANT: Regenerate the occupancy mask from the sampled visuals
            // If you don't do this, the mesher thinks the chunk is empty!
            chunk.occupancy = lod_visuals.1

            // chunk.occupancy.fill(0);
            // for i in 0..chunk.visuals.len() {
            //     if chunk.visuals[i] != 0 {
            //         let lx = i % 64;
            //         let ly = (i / 64) % 64;
            //         let lz = i / 4096;
            //         chunk.occupancy[lz * 64 + ly] |= 1 << lx;
            //     }
            // }
            // chunk.is_dirty = true;
        }

        self.hide_children_recursive(idx);

        // 5. LAST: Upload to GPU and Mesh
        self.texture_chunk_sparse(nx, ny, nz, depth);
        self.update_chunk_texture_sparse(nx, ny, nz, depth);
        self.mesh_chunk(nx, ny, nz, false, depth, true, false);
    }

    // -------------------------------------------------------------------------
    //  4. Meshing & Uploading
    // -------------------------------------------------------------------------

    #[func]
    pub fn mesh_chunk(
        &mut self,
        cx: i32,
        cy: i32,
        cz: i32,
        profile: bool,
        depth: u8,
        collision: bool,
        force_occlusion: bool,
    ) {
        let total_start = if profile { Some(Instant::now()) } else { None };

        let relative_voxel_size = self.voxel_size * (2.0_f32.powi(depth as i32));

        let scenario = self
            .base()
            .get_world_3d()
            .map(|w| w.get_scenario())
            .unwrap_or(Rid::Invalid);

        // --- 1. PREPARE DATA ---
        let chunk_ptr = if let Some(chunk) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
            chunk as *mut ChunkData
        } else {
            return;
        };
        let chunk = unsafe { &mut *chunk_ptr };
        let slices = [[0u64; 64]; 6];

        let mut std_verts = Vec::with_capacity(4000);
        let mut std_norms = Vec::with_capacity(4000);
        let mut std_indices = Vec::with_capacity(6000);
        let mut v_count = 0;

        // --- 2. GREEDY MESHING (CPU) ---
        let build_start = if profile { Some(Instant::now()) } else { None };
        unsafe {
            build!(u64, &chunk.occupancy, &slices, |n, x, y, z, h, w| {
                let (fx, fy, fz) = (
                    x as f32 * relative_voxel_size,
                    y as f32 * relative_voxel_size,
                    z as f32 * relative_voxel_size,
                );
                let (fh, fw) = (
                    h as f32 * relative_voxel_size,
                    w as f32 * relative_voxel_size,
                );
                let (p0, p1, p2, p3, norm) = match n {
                    1 => (
                        Vector3::new(fx + relative_voxel_size, fy, fz + fw),
                        Vector3::new(fx + relative_voxel_size, fy + fh, fz + fw),
                        Vector3::new(fx + relative_voxel_size, fy + fh, fz),
                        Vector3::new(fx + relative_voxel_size, fy, fz),
                        Vector3::RIGHT,
                    ),
                    0 => (
                        Vector3::new(fx, fy, fz),
                        Vector3::new(fx, fy + fh, fz),
                        Vector3::new(fx, fy + fh, fz + fw),
                        Vector3::new(fx, fy, fz + fw),
                        Vector3::LEFT,
                    ),
                    3 => (
                        Vector3::new(fz + fw, fx + relative_voxel_size, fy),
                        Vector3::new(fz + fw, fx + relative_voxel_size, fy + fh),
                        Vector3::new(fz, fx + relative_voxel_size, fy + fh),
                        Vector3::new(fz, fx + relative_voxel_size, fy),
                        Vector3::UP,
                    ),
                    2 => (
                        Vector3::new(fz, fx, fy),
                        Vector3::new(fz, fx, fy + fh),
                        Vector3::new(fz + fw, fx, fy + fh),
                        Vector3::new(fz + fw, fx, fy),
                        Vector3::DOWN,
                    ),
                    5 => (
                        Vector3::new(fy, fz + fw, fx + relative_voxel_size),
                        Vector3::new(fy + fh, fz + fw, fx + relative_voxel_size),
                        Vector3::new(fy + fh, fz, fx + relative_voxel_size),
                        Vector3::new(fy, fz, fx + relative_voxel_size),
                        Vector3::BACK,
                    ),
                    4 => (
                        Vector3::new(fy, fz, fx),
                        Vector3::new(fy + fh, fz, fx),
                        Vector3::new(fy + fh, fz + fw, fx),
                        Vector3::new(fy, fz + fw, fx),
                        Vector3::FORWARD,
                    ),
                    _ => unreachable!(),
                };
                std_verts.extend_from_slice(&[p0, p1, p2, p3]);
                std_norms.extend_from_slice(&[norm, norm, norm, norm]);
                std_indices.extend_from_slice(&[
                    v_count,
                    v_count + 1,
                    v_count + 2,
                    v_count,
                    v_count + 2,
                    v_count + 3,
                ]);
                v_count += 4;
            });
        }
        let build_time = build_start.map(|t| t.elapsed());

        if std_verts.is_empty() {
            return;
        }

        // --- 3. VISUALS (GPU) ---
        let gpu_start = if profile { Some(Instant::now()) } else { None };
        let mut rs = RenderingServer::singleton();
        //        let mut arrays = VariantArray::new();
        let mut arrays = Array::<Variant>::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        arrays.set(
            ArrayType::VERTEX.ord() as usize,
            &PackedVector3Array::from(std_verts.clone()).to_variant(),
        );
        arrays.set(
            ArrayType::NORMAL.ord() as usize,
            &PackedVector3Array::from(std_norms).to_variant(),
        );
        arrays.set(
            ArrayType::INDEX.ord() as usize,
            &PackedInt32Array::from(std_indices.clone()).to_variant(),
        );

        if !chunk.mesh_rid.is_valid() {
            chunk.mesh_rid = rs.mesh_create();
        }
        rs.mesh_clear(chunk.mesh_rid);
        rs.mesh_add_surface_from_arrays(chunk.mesh_rid, PrimitiveType::TRIANGLES, &arrays);
        //
        //
        //
        //
        //
        //

        if !chunk.instance_rid.is_valid() {
            chunk.instance_rid = rs.instance_create();
            rs.instance_set_scenario(chunk.instance_rid, scenario);
            rs.instance_set_base(chunk.instance_rid, chunk.mesh_rid);

            // --- MATERIAL SETUP ---
            // Reuse existing material or create new one
            if !chunk.material_rid.is_valid() {
                let mat_rid = rs.material_create();
                rs.material_set_shader(mat_rid, self.shader_rid);
                chunk.material_rid = mat_rid; // <--- CRITICAL: SAVE THIS!
            }

            // Bind material to instance
            rs.instance_geometry_set_material_override(chunk.instance_rid, chunk.material_rid);

            // Set position
            let chunk_world_step = 64.0 * relative_voxel_size;
            let transform = Transform3D::new(
                Basis::IDENTITY,
                Vector3::new(
                    cx as f32 * chunk_world_step,
                    cy as f32 * chunk_world_step,
                    cz as f32 * chunk_world_step,
                ),
            );
            rs.instance_set_transform(chunk.instance_rid, transform);
        }

        // 3. Ensure Texture exists and is bound
        // If the texture RID is invalid, generate it now.
        if !chunk.texture_rid.is_valid() {
            // We must drop the borrow on `chunk` to call self.update_chunk_texture
            // This requires a tiny bit of flow restructuring or just calling the logic inline.
            // For safety in Rust, it is often easier to Flag it and run updates in a separate pass,
            // but here is the logic:
        } else {
            // UNCOMMENT THIS
            // rs.instance_geometry_set_shader_parameter(
            //     chunk.instance_rid,
            //     StringName::from("voxel_data"),
            //     &Variant::from(chunk.texture_rid),
            // );

            // let origin = Vector3::new(
            //     cx as f32 * 64.0 * self.voxel_size,
            //     cy as f32 * 64.0 * self.voxel_size,
            //     cz as f32 * 64.0 * self.voxel_size,
            // );
            // rs.instance_geometry_set_shader_parameter(
            //     chunk.instance_rid,
            //     StringName::from("chunk_pos"),
            //     &Variant::from(origin),
            // );
        }

        //
        //
        //
        //
        //
        //
        //
        //
        //

        if !chunk.instance_rid.is_valid() {
            chunk.instance_rid = rs.instance_create();
            rs.instance_set_scenario(chunk.instance_rid, scenario);
            rs.instance_set_base(chunk.instance_rid, chunk.mesh_rid);

            if !chunk.material_rid.is_valid() {
                let mat_rid = rs.material_create();
                rs.material_set_shader(mat_rid, self.shader_rid);
                chunk.material_rid = mat_rid;
            }
            rs.instance_geometry_set_material_override(chunk.instance_rid, chunk.material_rid);
        }

        // --- BINDING THE NEW SPARSE TEXTURES ---
        // We must bind all three RIDs to the material for the shader to function
        if chunk.material_rid.is_valid() {
            // 1. Bitmask
            if chunk.bitmask_rid.is_valid() {
                rs.material_set_param(
                    chunk.material_rid,
                    &StringName::from("bitmask_tex"),
                    &Variant::from(chunk.bitmask_rid),
                );
            }
            // 2. Index Map
            if chunk.index_rid.is_valid() {
                rs.material_set_param(
                    chunk.material_rid,
                    &StringName::from("index_map_tex"),
                    &Variant::from(chunk.index_rid),
                );
            }
            // 3. Sparse Visuals
            if chunk.visuals_rid.is_valid() {
                rs.material_set_param(
                    chunk.material_rid,
                    &StringName::from("visual_data_tex"),
                    &Variant::from(chunk.visuals_rid),
                );
            }

            // 4. World Position (Required for local voxel lookup)
            let origin = Vector3::new(
                cx as f32 * 64.0 * self.voxel_size,
                cy as f32 * 64.0 * self.voxel_size,
                cz as f32 * 64.0 * self.voxel_size,
            );
            rs.material_set_param(
                chunk.material_rid,
                &StringName::from("voxel_size"),
                &Variant::from(relative_voxel_size),
            );
        }

        let chunk_world_step = 64.0 * self.voxel_size;
        let transform = Transform3D::new(
            Basis::IDENTITY,
            Vector3::new(
                (cx as f32 * chunk_world_step),
                (cy as f32 * chunk_world_step),
                (cz as f32 * chunk_world_step),
            ),
        );
        rs.instance_set_transform(chunk.instance_rid, transform);
        let gpu_time = gpu_start.map(|t| t.elapsed());

        let mut occlusion_check = false;
        if self.enable_occlusion == true {
            if chunk.calculate_density() > self.occlusion__density_threshold
                || force_occlusion == true
            {
                occlusion_check = true;
            }
        }

        let col_start = if profile { Some(Instant::now()) } else { None };
        if collision == true || occlusion_check == true {
            // --- 4. COLLISION (Physics) ---

            // Use a temporary Vec to collect vertices.
            // A downsampled 32x32x32 grid significantly reduces vertex count.
            let mut col_verts = Vec::with_capacity(2000);

            crate::vxl_mesher::build_collider_fast(&chunk.occupancy, |face, x, y, z, h, d| {
                // Scale local 32-grid coords to 64-grid world space
                let (fx, fy, fz) = (
                    x as f32 * relative_voxel_size * 2.0,
                    y as f32 * relative_voxel_size * 2.0,
                    z as f32 * relative_voxel_size * 2.0,
                );
                let (fh, fd) = (
                    h as f32 * relative_voxel_size * 2.0,
                    d as f32 * relative_voxel_size * 2.0,
                );

                // Match the exact vertex order from the visual mesh
                // face: 0=LEFT, 1=RIGHT, 2=DOWN, 3=UP, 4=FORWARD, 5=BACK
                let (p0, p1, p2, p3) = match face {
                    1 => (
                        // RIGHT (X+)
                        Vector3::new(fx + relative_voxel_size * 2.0, fy, fz + fd),
                        Vector3::new(fx + relative_voxel_size * 2.0, fy + fh, fz + fd),
                        Vector3::new(fx + relative_voxel_size * 2.0, fy + fh, fz),
                        Vector3::new(fx + relative_voxel_size * 2.0, fy, fz),
                    ),
                    0 => (
                        // LEFT (X-)
                        Vector3::new(fx, fy, fz),
                        Vector3::new(fx, fy + fh, fz),
                        Vector3::new(fx, fy + fh, fz + fd),
                        Vector3::new(fx, fy, fz + fd),
                    ),
                    3 => (
                        // UP (Y+)
                        Vector3::new(fz + fd, fx + relative_voxel_size * 2.0, fy),
                        Vector3::new(fz + fd, fx + relative_voxel_size * 2.0, fy + fh),
                        Vector3::new(fz, fx + relative_voxel_size * 2.0, fy + fh),
                        Vector3::new(fz, fx + relative_voxel_size * 2.0, fy),
                    ),
                    2 => (
                        // DOWN (Y-)
                        Vector3::new(fz, fx, fy),
                        Vector3::new(fz, fx, fy + fh),
                        Vector3::new(fz + fd, fx, fy + fh),
                        Vector3::new(fz + fd, fx, fy),
                    ),
                    5 => (
                        // BACK (Z+)
                        Vector3::new(fy, fz + fd, fx + relative_voxel_size * 2.0),
                        Vector3::new(fy + fh, fz + fd, fx + relative_voxel_size * 2.0),
                        Vector3::new(fy + fh, fz, fx + relative_voxel_size * 2.0),
                        Vector3::new(fy, fz, fx + relative_voxel_size * 2.0),
                    ),
                    4 => (
                        // FORWARD (Z-)
                        Vector3::new(fy, fz, fx),
                        Vector3::new(fy + fh, fz, fx),
                        Vector3::new(fy + fh, fz + fd, fx),
                        Vector3::new(fy, fz + fd, fx),
                    ),
                    _ => unreachable!(),
                };

                // Push 6 vertices (2 triangles) per greedy quad
                col_verts.extend_from_slice(&[p0, p1, p2, p0, p2, p3]);
            });

            if collision == true {
                if !col_verts.is_empty() {
                    let mut shape = ConcavePolygonShape3D::new_gd();
                    shape.set_faces(&PackedVector3Array::from(col_verts.clone()));

                    if chunk.collider.is_none() {
                        let node = CollisionShape3D::new_alloc();
                        self.base_mut().add_child(&node);
                        chunk.collider = Some(node);
                    }

                    if let Some(node) = &mut chunk.collider {
                        node.set_shape(&shape);
                        // The visual mesh and collision mesh share the same origin
                        node.set_position(transform.origin);
                    }
                } else if let Some(node) = &mut chunk.collider {
                    // Clear collision if the chunk became empty
                    node.set_shape(&ConcavePolygonShape3D::new_gd());
                }
            }

            if occlusion_check == true {
                if !col_verts.is_empty() {
                    // 1. Ensure the Occluder resource exists
                    if !chunk.occluder_rid.is_valid() {
                        chunk.occluder_rid = rs.occluder_create();
                    }

                    // 2. The RenderingServer requires an index buffer for occluders.
                    // Since our col_verts is an unindexed triangle list, we just map 0..N
                    let indices: Vec<i32> = (0..col_verts.len() as i32).collect();

                    // Upload the mesh data to the occluder
                    rs.occluder_set_mesh(
                        chunk.occluder_rid,
                        &PackedVector3Array::from(col_verts),
                        &PackedInt32Array::from(indices),
                    );

                    // 3. Ensure the Occluder is instanced in the world
                    if !chunk.occluder_instance_rid.is_valid() {
                        chunk.occluder_instance_rid = rs.instance_create();
                        rs.instance_set_scenario(chunk.occluder_instance_rid, scenario);
                        rs.instance_set_base(chunk.occluder_instance_rid, chunk.occluder_rid);
                    }

                    // Transform matches the visual chunk
                    rs.instance_set_transform(chunk.occluder_instance_rid, transform);
                } else if chunk.occluder_rid.is_valid() {
                    // Clear the occluder mesh to prevent ghost culling if the chunk becomes empty
                    rs.occluder_set_mesh(
                        chunk.occluder_rid,
                        &PackedVector3Array::new(),
                        &PackedInt32Array::new(),
                    );
                }
            }
        }
        let col_time = col_start.map(|t| t.elapsed());
        // --- 5. REPORT ---
        if let Some(total_t) = total_start {
            godot_print!(
                "[Profile] Chunk ({},{},{}): Total {:?} | Build {:?} | GPU {:?} | (if enabled) Collision {:?}",
                cx,
                cy,
                cz,
                total_t.elapsed(),
                build_time.unwrap(),
                gpu_time.unwrap(),
                col_time.unwrap()
            );
        }
    }

    // -------------------------------------------------------------------------
    //  Texture Management
    // -------------------------------------------------------------------------

    // #[func]
    // pub fn update_chunk_texture_fast(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
    //     let mut rs = RenderingServer::singleton();

    //     // 1. Get chunk and exit early if missing
    //     // let chunk = match self.chunks.get_mut(&key) {
    //     //     Some(c) => c,
    //     //     None => return,
    //     // };
    //     let chunk = self.octree.get_node_data_mut(cx, cy, cz, depth).unwrap();

    //     // 2. Prepare the Image Array
    //     // We pre-allocate the capacity to avoid re-allocations inside the loop
    //     let mut images = Array::new();

    //     // Use a temporary stack buffer for byte conversion (64*64*2 bytes = 8192)
    //     // This avoids 64 heap allocations of PackedByteArray per chunk.
    //     let mut temp_bytes = [0u8; 8192];

    //     for z in 0..64 {
    //         let start = z * 4096;
    //         let slice_u32s = &chunk.visuals[start..start + 4096];

    //         // Optimized bit-packing loop: chunks_exact_mut helps the compiler vectorize
    //         for (pixel, dst) in slice_u32s.iter().zip(temp_bytes.chunks_exact_mut(2)) {
    //             // Extracts RRRRGGGG into high and BBBBEEMT into low
    //             dst[0] = (*pixel >> 8) as u8;
    //             dst[1] = *pixel as u8;
    //         }

    //         // Create image from the stack buffer.
    //         // Note: Image::create_from_data copies the buffer into Godot-managed memory.
    //         if let Some(img) = Image::create_from_data(
    //             64,
    //             64,
    //             false,
    //             Format::RG8,
    //             &PackedByteArray::from_iter(temp_bytes),
    //         ) {
    //             images.push(&img);
    //         }
    //     }

    //     // 3. Smart GPU Upload
    //     if chunk.texture_rid.is_valid() {
    //         // Significantly faster: updates existing texture memory without recreation
    //         rs.texture_3d_update(chunk.texture_rid, &images);
    //     } else {
    //         // First time creation
    //         let texture_rid = rs.texture_3d_create(Format::RG8, 64, 64, 64, false, &images);
    //         chunk.texture_rid = texture_rid;

    //         if chunk.material_rid.is_valid() {
    //             // Cache the StringName if this is called frequently
    //             static PARAM_NAME: &str = "voxel_data";
    //             rs.material_set_param(
    //                 chunk.material_rid,
    //                 &StringName::from(PARAM_NAME),
    //                 &Variant::from(texture_rid),
    //             );
    //         }
    //     }

    //     chunk.is_dirty = false;
    // }

    // #[func]
    // pub fn update_chunk_texture_32(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
    //     let mut rs = RenderingServer::singleton();

    //     let chunk = if let Some(data) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
    //         data
    //     } else {
    //         return;
    //     };

    //     // 1. Prepare the Array of 64 Slices
    //     let mut images = Array::new();

    //     for z in 0..64 {
    //         let start_idx = z * 4096;
    //         let end_idx = start_idx + 4096;
    //         let slice_u32s = &self.gpu_buffer[start_idx..end_idx];

    //         // 2. Fast Memory Copy for the slice
    //         // Create the PackedByteArray from an iterator of the byte slice
    //         let byte_slice = unsafe {
    //             let ptr = slice_u32s.as_ptr() as *const u8;
    //             let byte_len = 4096 * 4;
    //             std::slice::from_raw_parts(ptr, byte_len)
    //         };

    //         // Use from_iter to create the Godot array efficiently
    //         let slice_bytes = PackedByteArray::from_iter(byte_slice.iter().copied());

    //         // 3. Create a 64x64 Image for this layer
    //         // Note: Image::create_from_data returns an Option/Result depending on your version
    //         let image = Image::create_from_data(64, 64, false, Format::RG16, &slice_bytes);

    //         if let Some(img) = image {
    //             images.push(&img);
    //         }
    //     }
    //     // 4. Update or Create the Texture3D
    //     if chunk.texture_rid.is_valid() {
    //         // Pass the array of 64 images to update the layers
    //         rs.texture_3d_update(chunk.texture_rid, &images);
    //     } else {
    //         let texture_rid = rs.texture_3d_create(Format::RG16, 64, 64, 64, false, &images);
    //         chunk.texture_rid = texture_rid;

    //         if chunk.material_rid.is_valid() {
    //             static PARAM_NAME: &str = "voxel_data";
    //             rs.material_set_param(
    //                 chunk.material_rid,
    //                 &StringName::from(PARAM_NAME),
    //                 &Variant::from(texture_rid),
    //             );
    //         }
    //     }

    //     chunk.is_dirty = false;
    // }

    #[func]
    pub fn update_chunk_texture_sparse(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
        let mut rs = RenderingServer::singleton();

        let chunk = if let Some(data) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
            data
        } else {
            return;
        };

        // --- 1. BITMASK (64x64, 64-bit per pixel -> RGBA16) ---
        let bitmask_bytes = unsafe {
            let ptr = self.gpu_upload_data.bitmask.as_ptr() as *const u8;
            let len = self.gpu_upload_data.bitmask.len() * 8;
            std::slice::from_raw_parts(ptr, len)
        };
        let bitmask_packed = PackedByteArray::from_iter(bitmask_bytes.iter().copied());
        let bitmask_img = Image::create_from_data(64, 64, false, Format::RGBA16, &bitmask_packed)
            .expect("Failed to create bitmask image");

        // --- 2. INDEX MAP (64x64, 32-bit per pixel -> RG16) ---
        let index_bytes = unsafe {
            let ptr = self.gpu_upload_data.index_map.as_ptr() as *const u8;
            let len = self.gpu_upload_data.index_map.len() * 4;
            std::slice::from_raw_parts(ptr, len)
        };
        let index_packed = PackedByteArray::from_iter(index_bytes.iter().copied());
        let index_img = Image::create_from_data(64, 64, false, Format::RG16, &index_packed)
            .expect("Failed to create index map image");

        // --- 3. SPARSE VISUALS (Nx1, 32-bit per voxel -> RG16) ---
        let voxel_count = self.gpu_upload_data.voxel_count;

        // Update or Create Bitmask Texture
        if chunk.bitmask_rid.is_valid() {
            rs.texture_2d_update(chunk.bitmask_rid, &bitmask_img, 0);
        } else {
            chunk.bitmask_rid = rs.texture_2d_create(&bitmask_img);
        }

        // Update or Create Index Map Texture
        if chunk.index_rid.is_valid() {
            rs.texture_2d_update(chunk.index_rid, &index_img, 0);
        } else {
            chunk.index_rid = rs.texture_2d_create(&index_img);
        }

        // Handle Sparse Visuals
        if voxel_count > 0 {
            let visual_bytes = unsafe {
                let ptr = self.gpu_upload_data.sparse_visuals.as_ptr() as *const u8;
                let len = voxel_count as usize * 4;
                std::slice::from_raw_parts(ptr, len)
            };
            let visual_packed = PackedByteArray::from_iter(visual_bytes.iter().copied());

            // Note: Using RG16 for 32-bit voxel data
            let visual_img =
                Image::create_from_data(voxel_count as i32, 1, false, Format::RG16, &visual_packed)
                    .expect("Failed to create sparse visuals image");

            // We recreate the texture if the size changed (common for sparse data)
            // because texture_2d_update requires matching dimensions.
            if chunk.visuals_rid.is_valid() {
                rs.free_rid(chunk.visuals_rid);
            }
            chunk.visuals_rid = rs.texture_2d_create(&visual_img);
        }

        // --- 4. MATERIAL PARAMETERS ---
        if chunk.material_rid.is_valid() {
            // Ensure your shader uses: uniform sampler2D bitmask_tex, index_map_tex, visual_data_tex;
            rs.material_set_param(
                chunk.material_rid,
                &StringName::from("bitmask_tex"),
                &Variant::from(chunk.bitmask_rid),
            );
            rs.material_set_param(
                chunk.material_rid,
                &StringName::from("index_map_tex"),
                &Variant::from(chunk.index_rid),
            );
            rs.material_set_param(
                chunk.material_rid,
                &StringName::from("visual_data_tex"),
                &Variant::from(chunk.visuals_rid),
            );
        }

        chunk.is_dirty = false;
    }

    // #[func]
    // pub fn update_chunk_texture(&mut self, cx: i32, cy: i32, cz: i32) {
    //     //let key = Self::coords_to_key(cx, cy, cz);
    //     let mut rs = RenderingServer::singleton();

    //     // Access the chunk
    //     if let Some(chunk) = self.octree.get_leaf_data_mut(cx, cy, cz) {
    //         // --- 1. PACK DATA (CPU -> Bytes) ---
    //         let mut images = Array::new();

    //         for z in 0..64usize {
    //             let start = z * 4096;
    //             let end = start + 4096;
    //             let slice_u32s: &[u32] = &chunk.visuals[start..end];

    //             let mut bytes = PackedByteArray::new();
    //             // Pre-allocate: 64 * 64 * 2 bytes = 8192 bytes
    //             bytes.resize(8192);

    //             // Get a mutable view of the byte data
    //             let mut byte_slice = bytes.as_mut_slice();

    //             for (i, &pixel) in slice_u32s.iter().enumerate() {
    //                 let high = ((pixel >> 8) & 0xFF) as u8;
    //                 let low = (pixel & 0xFF) as u8;

    //                 byte_slice[i * 2] = high;
    //                 byte_slice[i * 2 + 1] = low;
    //             }

    //             if let Some(img) = Image::create_from_data(64, 64, false, Format::RG8, &bytes) {
    //                 images.push(&img);
    //             }
    //         }

    //         // --- 2. UPLOAD (Server API) ---

    //         // Free old RID if it exists
    //         if chunk.texture_rid.is_valid() {
    //             rs.free_rid(chunk.texture_rid);
    //         }

    //         // Create new Texture RID directly on Server
    //         let texture_rid = rs.texture_3d_create(Format::RG8, 64, 64, 64, false, &images);

    //         chunk.texture_rid = texture_rid;

    //         // --- 3. BIND ---
    //         // Only bind if the material actually exists (Meshing has run)
    //         if chunk.material_rid.is_valid() {
    //             rs.material_set_param(
    //                 chunk.material_rid,
    //                 &StringName::from("voxel_data"),
    //                 &Variant::from(texture_rid),
    //             );
    //         }
    //     }
    // }

    // -------------------------------------------------------------------------
    //  Helpers
    // -------------------------------------------------------------------------

    // Helper to remove children's meshes from Godot when LOD takes over
    fn hide_children_recursive(&mut self, parent_idx: u32) {
        let mut rs = RenderingServer::singleton();
        let children = self.octree.node_pool[parent_idx as usize].children;

        for &child_idx in children.iter() {
            if child_idx == u32::MAX {
                continue;
            }

            self.hide_children_recursive(child_idx);

            // Take data out of children to free RIDs and stop rendering
            if let Some(mut chunk) = self.octree.node_pool[child_idx as usize].chunk.take() {
                if chunk.instance_rid.is_valid() {
                    rs.free_rid(chunk.instance_rid);
                }
                if chunk.mesh_rid.is_valid() {
                    rs.free_rid(chunk.mesh_rid);
                }
                if chunk.texture_rid.is_valid() {
                    rs.free_rid(chunk.texture_rid);
                }
                if let Some(col) = &mut chunk.collider {
                    col.queue_free();
                }
            }
        }
    }

    /// Sets a single voxel at global world coordinates.
    #[func]
    pub fn set_voxel_bit(&mut self, gx: i32, gy: i32, gz: i32, on: bool) {
        let cx = gx.div_euclid(64);
        let cy = gy.div_euclid(64);
        let cz = gz.div_euclid(64);

        let lx = gx.rem_euclid(64) as usize;
        let ly = gy.rem_euclid(64) as usize;
        let lz = gz.rem_euclid(64) as usize;

        // We only edit if the chunk exists.
        // Logic could be added here to auto-create chunks if you want.
        if let Some(chunk) = self.octree.get_leaf_data_mut(cx, cy, cz) {
            let index = lz * 64 + ly;
            if on {
                chunk.occupancy[index] |= 1 << lx;
            } else {
                chunk.occupancy[index] &= !(1 << lx);
            }
        }
    }

    /// Gets a single voxel state from global world coordinates.
    #[func]
    pub fn get_voxel_bit(&self, gx: i32, gy: i32, gz: i32) -> bool {
        let cx = gx.div_euclid(64);
        let cy = gy.div_euclid(64);
        let cz = gz.div_euclid(64);

        if let Some(chunk) = self.octree.get_leaf_data(cx, cy, cz) {
            let lx = gx.rem_euclid(64) as usize;
            let ly = gy.rem_euclid(64) as usize;
            let lz = gz.rem_euclid(64) as usize;

            return (chunk.occupancy[lz * 64 + ly] & (1 << lx)) != 0;
        }
        false
    }

    #[inline(always)]
    fn pack_visuals(
        material: u8,
        r: u32,
        g: u32,
        b: u32,
        emit: bool, // Changed to bool for the 4-bit toggle
        metal: bool,
        trans: bool,
        rough: bool, // Added roughness toggle
        has_tex: bool,
    ) -> u32 {
        // 1. Pack 15-bit Color (5 bits per channel)
        let r_part = (r >> 3) & 0x1F;
        let g_part = (g >> 3) & 0x1F;
        let b_part = (b >> 3) & 0x1F;
        let color_15b = (r_part << 10) | (g_part << 5) | b_part;

        // 2. Texture Flag (Bit 15)
        let tex_bit = if has_tex { 1u32 << 15 } else { 0u32 };

        // 3. Material ID (Bits 16-23)
        let mat_part = (material as u32 & 0xFF) << 16;

        // 4. Extras Toggles (Bits 24-27)
        let mut extras = 0u32;
        if emit {
            extras |= 1 << 0;
        }
        if metal {
            extras |= 1 << 1;
        }
        if trans {
            extras |= 1 << 2;
        }
        if rough {
            extras |= 1 << 3;
        }
        let extras_part = extras << 24;

        // Result: [4-bit Empty][4-bit Extras][8-bit Material][1-bit Tex][15-bit Color]
        color_15b | tex_bit | mat_part | extras_part
    }

    pub fn pack_gpu_voxel(
        color_15b: u32,
        nx_4b: u32,
        ny_4b: u32,
        nz_4b: u32,
        emit: bool,
        metal: bool,
        trans: bool,
        rough: bool,
    ) -> u32 {
        let mut packed = color_15b & 0x7FFF; // 0-14

        // Normals (15-26)
        let normal_part = (nx_4b & 0xF) | ((ny_4b & 0xF) << 4) | ((nz_4b & 0xF) << 8);
        packed |= normal_part << 15;

        // Flags (27-30)
        if emit {
            packed |= 1 << 27;
        }
        if metal {
            packed |= 1 << 28;
        }
        if trans {
            packed |= 1 << 29;
        }
        if rough {
            packed |= 1 << 30;
        }

        packed
    }

    #[inline(always)]
    fn set_voxel_material(packed_visual: u32, new_mat_id: u8) -> u32 {
        // 1. Create a mask to clear bits 16-23 (0xFF0000)
        // 2. Preserve bits 0-15 (Colors/Attributes) and 24-31 (Extra/Hardness)
        let preserved_data = packed_visual & !0x00FF0000;

        // 3. Shift the new material ID into place and combine
        preserved_data | ((new_mat_id as u32 & 0xFF) << 16)
    }

    // #[inline(always)]
    // fn pack_visuals(r: u32, g: u32, b: u32, emit: u32, metal: bool, trans: bool) -> u32 {
    //     let m = if metal { 1 } else { 0 };
    //     let t = if trans { 1 } else { 0 };

    //     // RRRR GGGG BBBB EEMT
    //     ((r & 0xF) << 12) | ((g & 0xF) << 8) | ((b & 0xF) << 4) | ((emit & 0x3) << 2) | (m << 1) | t
    // }

    /// Helper: Write to both the fast mask AND the detailed array
    fn set_voxel_internal(
        &mut self,
        chunk: &mut ChunkData,
        lx: usize,
        ly: usize,
        lz: usize,
        data: u32,
    ) {
        let index = lz * 4096 + ly * 64 + lx; // Standard flat index

        // 1. Update Detailed Data
        chunk.visuals[index] = data;

        // 2. Update Fast Mask
        // Note: The mask uses a different indexing strategy (y is inner loop usually in my examples)
        // Adjust this math to match your specific 'morton' or 'linear' layout in occupancy.
        let mask_idx = lz * 64 + ly;
        if data != 0 {
            chunk.occupancy[mask_idx] |= 1 << lx;
        } else {
            chunk.occupancy[mask_idx] &= !(1 << lx);
        }

        chunk.is_dirty = true;
    }

    // Simple Tri-linear interpolation on the 4x4x4 (size 64 flat) coarse grid
    // coordinates x,y,z should be in range 0.0 .. 3.0
    fn trilinear_interp(x: f32, y: f32, z: f32, data: &[f32; 64]) -> f32 {
        // Clamp to valid range to prevent panic on edge sampling
        let x = x.clamp(0.0, 2.999);
        let y = y.clamp(0.0, 2.999);
        let z = z.clamp(0.0, 2.999);

        let x0 = x as usize;
        let y0 = y as usize;
        let z0 = z as usize;

        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let xd = x - x0 as f32;
        let yd = y - y0 as f32;
        let zd = z - z0 as f32;

        // Fetch 8 neighbors
        // Index = z * 16 + y * 4 + x
        let i000 = (z0 * 16) + (y0 * 4) + x0;
        let i001 = (z0 * 16) + (y0 * 4) + x1;
        let i010 = (z0 * 16) + (y1 * 4) + x0;
        let i011 = (z0 * 16) + (y1 * 4) + x1;
        let i100 = (z1 * 16) + (y0 * 4) + x0;
        let i101 = (z1 * 16) + (y0 * 4) + x1;
        let i110 = (z1 * 16) + (y1 * 4) + x0;
        let i111 = (z1 * 16) + (y1 * 4) + x1;

        let c00 = data[i000] * (1.0 - xd) + data[i001] * xd;
        let c01 = data[i010] * (1.0 - xd) + data[i011] * xd;
        let c10 = data[i100] * (1.0 - xd) + data[i101] * xd;
        let c11 = data[i110] * (1.0 - xd) + data[i111] * xd;

        let c0 = c00 * (1.0 - yd) + c01 * yd;
        let c1 = c10 * (1.0 - yd) + c11 * yd;

        c0 * (1.0 - zd) + c1 * zd
    }

    //
    //  Terrain Generation Functions
    //
    //

    /// Checks if a chunk contains the surface (Mixed) to decide if we should generate it.
    #[func]
    pub fn is_chunk_mixed(&self, cx: i32, cy: i32, cz: i32, depth: i32) -> bool {
        if let Some(tm) = &self.terrain_manager {
            let tm_bind = tm.bind();
            if let Some(chunk_meta) = tm_bind
                .chunk_metadata
                .get(&Vector4i::new(cx, cy, cz, depth))
            {
                return chunk_meta.state == ChunkState::Mixed;
            }
        }
        // If no metadata exists, assume it's empty or handle as needed
        false
    }

    #[func]
    pub fn generate_chunk_3d(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) -> bool {
        let start = Instant::now();
        let (coarse_density, material_ids) = {
            // Clone the Gd handle so we have an owned copy to call bind_mut() on
            if let Some(mut tm) = self.terrain_manager.clone() {
                let mut tm_bind = tm.bind_mut(); // This should now work
                if let Some(meta) = tm_bind
                    .chunk_metadata
                    .get(&Vector4i::new(cx, cy, cz, depth))
                {
                    if meta.state != ChunkState::Mixed {
                        return false;
                    }
                    // Return clones of the data we need so we can drop the borrow
                    (meta.coarse_density.clone(), meta.material_ids.clone())
                } else {
                    return false;
                }
            } else {
                return false;
            }
        };

        let macro_start = Instant::now();
        if let Some(chunk) = self.octree.get_node_data_mut(cx, cy, cz, depth as u8) {
            chunk.occupancy.fill(0);

            // With 8 points, we have 7 segments.
            // However, our padding means segment 0 and segment 6 are mostly outside the chunk.
            // We iterate all 7 segments, but the internal loop will clamp to 0..64.
            for cz_idx in 0..7 {
                for cy_idx in 0..7 {
                    for cx_idx in 0..7 {
                        Self::generate_macro_cell(
                            cx_idx,
                            cy_idx,
                            cz_idx,
                            depth as usize,
                            &coarse_density,
                            &material_ids,
                            chunk,
                        );
                    }
                }
            }

            chunk.is_dirty = true;

            // CLEANUP: Free memory from the Resource after generation
            if let Some(mut tm) = self.terrain_manager.clone() {
                tm.bind_mut().clear_coarse_data(cx, cy, cz, depth);
            }

            let macro_time = macro_start.elapsed();

            godot_print!(
                "Chunk [{cx},{cy},{cz}] -> Macro: {:?}, Total: {:?}",
                macro_time,
                start.elapsed()
            );

            return true;
        }

        false
    }

    fn generate_macro_cell(
        cx: usize,
        cy: usize,
        cz: usize,
        depth: usize,
        coarse: &[f32; 512],
        material_ids: &[u8; 512],
        chunk: &mut ChunkData,
    ) {
        let step = 12.8;
        let inv_step = 1.0 / step;

        // Ranges for this macro cell
        let x_start_f = (cx as f32 - 1.0) * step;
        let y_start_f = (cy as f32 - 1.0) * step;
        let z_start_f = (cz as f32 - 1.0) * step;

        let x_start = (x_start_f.max(0.0).round() as usize).min(64);
        let x_end = (cx as f32 * step).max(0.0).round().min(64.0) as usize;
        let y_start = (y_start_f.max(0.0).round() as usize).min(64);
        let y_end = (cy as f32 * step).max(0.0).round().min(64.0) as usize;
        let z_start = (z_start_f.max(0.0).round() as usize).min(64);
        let z_end = (cz as f32 * step).max(0.0).round().min(64.0) as usize;

        if x_start >= x_end || y_start >= y_end || z_start >= z_end {
            return;
        }

        let material = material_ids[(cz * 64) + (cy * 8) + cx];
        let packed_color =
            Self::pack_visuals(material, 10, 12, 10, false, false, false, false, true);

        // Fetch the 8 corners of the macro cell
        let p000 = coarse[(cz * 64) + (cy * 8) + cx];
        let p100 = coarse[(cz * 64) + (cy * 8) + (cx + 1)];
        let p010 = coarse[(cz * 64) + ((cy + 1) * 8) + cx];
        let p110 = coarse[(cz * 64) + ((cy + 1) * 8) + (cx + 1)];
        let p001 = coarse[((cz + 1) * 64) + (cy * 8) + cx];
        let p101 = coarse[((cz + 1) * 64) + (cy * 8) + (cx + 1)];
        let p011 = coarse[((cz + 1) * 64) + ((cy + 1) * 8) + cx];
        let p111 = coarse[((cz + 1) * 64) + ((cy + 1) * 8) + (cx + 1)];

        for z in z_start..z_end {
            let z_f = (z as f32 - z_start_f) * inv_step;

            // 1. Optimization: Pre-calculate Z-interpolated values (Move 4 lerps out of X loop)
            let p00_z = Self::lerp(p000, p001, z_f);
            let p10_z = Self::lerp(p100, p101, z_f);
            let p01_z = Self::lerp(p010, p011, z_f);
            let p11_z = Self::lerp(p110, p111, z_f);

            let z_visual_off = z * 4096;
            let z_occ_off = z * 64;

            for x in x_start..x_end {
                let x_f = (x as f32 - x_start_f) * inv_step;

                // 2. Optimization: Calculate column bounds (Final bilinear lerp)
                let col_bot = Self::lerp(p00_z, p10_z, x_f);
                let col_top = Self::lerp(p01_z, p11_z, x_f);

                // density(y) = d_start + (y_local) * d_step
                let d_start =
                    col_bot + (y_start as f32 - y_start_f) * (col_top - col_bot) * inv_step;
                let d_step = (col_top - col_bot) * inv_step;

                // 3. Optimization: Solve for y range where density > 0 instead of voxel-wise checks
                let mut y_fill_start = y_end;
                let mut y_fill_end = y_end;

                if d_step.abs() < 1e-6 {
                    if d_start > 0.0 {
                        y_fill_start = y_start;
                        y_fill_end = y_end;
                    }
                } else {
                    let i_cross = -d_start / d_step; // The relative index within y_start..y_end where crossing happens

                    if d_step > 0.0 {
                        // Density is increasing: Positive for indices > i_cross
                        let first_i = (i_cross + 1e-5).ceil() as i32;
                        y_fill_start = (y_start as i32 + first_i)
                            .max(y_start as i32)
                            .min(y_end as i32) as usize;
                        y_fill_end = y_end;
                    } else {
                        // Density is decreasing: Positive for indices < i_cross
                        let last_i = (i_cross - 1e-5).floor() as i32;
                        y_fill_start = y_start;
                        y_fill_end = (y_start as i32 + last_i + 1)
                            .max(y_start as i32)
                            .min(y_end as i32) as usize;
                    }
                }

                if y_fill_start < y_fill_end {
                    // 4. Optimization: Bulk write Visuals
                    let base_idx = z_visual_off + (x * 64);
                    chunk.visuals[base_idx + y_fill_start..base_idx + y_fill_end]
                        .fill(packed_color);

                    // 5. Optimization: Bulk write Occupancy (Bitmask)
                    let len = y_fill_end - y_fill_start;
                    let mask = if len >= 64 {
                        !0u64
                    } else {
                        ((1u64 << len) - 1) << y_fill_start
                    };
                    chunk.occupancy[z_occ_off + x] |= mask;
                }
            }
        }
    }

    #[inline(always)]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    /////////////
    //
    //  High Level Functions
    //
    // //////////

    #[func]
    pub fn set_voxel_full(
        &mut self,
        gx: i32,
        gy: i32,
        gz: i32,
        material: u32,
        r: u32,
        g: u32,
        b: u32,
        emit: bool,
        metal: bool,
        trans: bool,
        roughness: bool,
        is_active: bool,
        has_tex: bool,
        paint_only: bool,
    ) {
        if paint_only && !self.get_voxel_bit(gx, gy, gz) {
            return;
        }

        let cx = gx.div_euclid(64);
        let cy = gy.div_euclid(64);
        let cz = gz.div_euclid(64);

        let lx = gx.rem_euclid(64) as usize;
        let ly = gy.rem_euclid(64) as usize;
        let lz = gz.rem_euclid(64) as usize;

        // 1. Get the index from the octree
        if let Some(node_idx) = self.octree.get_leaf_safe(cx, cy, cz, true) {
            // 2. Access the actual node and its chunk from the manager's pool
            let node = &mut self.octree.node_pool[node_idx as usize];

            // 3. Ensure the chunk exists in that node
            if node.chunk.is_none() {
                node.chunk = Some(ChunkData::new());
            }

            if let Some(chunk) = &mut node.chunk {
                let visual_index = (lz * 4096) + (ly * 64) + lx;
                let occupancy_index = (lz * 64) + ly;

                if is_active {
                    chunk.visuals[visual_index] = Self::pack_visuals(
                        material as u8,
                        r,
                        g,
                        b,
                        emit,
                        metal,
                        trans,
                        roughness,
                        has_tex,
                    );
                    if !paint_only {
                        chunk.occupancy[occupancy_index] |= 1 << lx;
                    }
                } else if !paint_only {
                    chunk.visuals[visual_index] = 0;
                    chunk.occupancy[occupancy_index] &= !(1 << lx);
                }

                chunk.is_dirty = true;
            }
        }
    }

    #[func]
    pub fn texture_chunk_sparse(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
        let terrain_manager = if let Some(tm) = &self.terrain_manager {
            tm.bind()
        } else {
            return;
        };

        let dictionary = &terrain_manager.baked_registry.voxel_dictionary;

        // 1. Get the chunk data from the octree
        let chunk = if let Some(data) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
            data
        } else {
            return;
        };

        // 2. Prepare the GpuUploadData for filling
        // We clear the previous sparse visuals and reset the voxel count
        self.gpu_upload_data.sparse_visuals.clear();
        self.gpu_upload_data.voxel_count = 0;

        let world_base_x = cx * 64;
        let world_base_y = cy * 64;
        let world_base_z = cz * 64;

        // Iterate through each 64-voxel column (along X) in the 64x64 grid (Z, Y)
        for z in 0..64 {
            let z_mask_off = z * 64;
            let z_vis_off = z * 4096;

            let zm_off = if z > 0 { (z - 1) * 64 } else { z_mask_off };
            let zp_off = if z < 63 { (z + 1) * 64 } else { z_mask_off };
            let is_on_z_boundary = z == 0 || z == 63;

            for y in 0..64 {
                let col_idx = z_mask_off + y;

                // --- 3. FILL INDEX MAP (Prefix Sum) ---
                // Store the total number of voxels processed BEFORE this column.
                // On the GPU, reading this value gives the starting offset into sparse_visuals.
                self.gpu_upload_data.index_map[col_idx] = self.gpu_upload_data.voxel_count;

                let mask = chunk.occupancy[col_idx];
                if mask == 0 {
                    self.gpu_upload_data.bitmask[col_idx] = 0;
                    continue;
                }

                let ym_mask = if y > 0 {
                    chunk.occupancy[z_mask_off + y - 1]
                } else {
                    mask
                };
                let yp_mask = if y < 63 {
                    chunk.occupancy[z_mask_off + y + 1]
                } else {
                    mask
                };
                let zm_mask = chunk.occupancy[zm_off + y];
                let zp_mask = chunk.occupancy[zp_off + y];

                // --- 4. EXPOSURE LOGIC ---
                // Identify voxels on the surface (exposed to air) to skip hidden internal data.
                let hidden_by_mirror =
                    (mask << 1) & (mask >> 1) & ym_mask & yp_mask & zm_mask & zp_mask;
                let is_on_x_boundary = mask & 0x8000000000000001;
                let is_on_y_boundary = y == 0 || y == 63;

                let mut exposed_mask = mask & !hidden_by_mirror;
                if is_on_y_boundary || is_on_z_boundary || (is_on_x_boundary != 0) {
                    let edge_bits = if is_on_y_boundary || is_on_z_boundary {
                        mask
                    } else {
                        is_on_x_boundary
                    };
                    exposed_mask |= (mask & edge_bits);
                }

                // --- 5. STORE BITMASK ---
                // Save the bitmask for this column to the upload struct.
                self.gpu_upload_data.bitmask[col_idx] = exposed_mask;

                // --- 6. PROCESS SURFACE VOXELS ---
                let mut bits = exposed_mask;
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    let vis_idx = z_vis_off + (y * 64) + x;

                    // CPU Visual Data Extraction
                    let cpu_packed = chunk.visuals[vis_idx];
                    let mat_id = (cpu_packed >> 16) & 0xFF;
                    let has_tex = (cpu_packed >> 15) & 1 == 1;

                    let mut final_gpu_val: u32;

                    if has_tex {
                        // TEXTURED PATH: Handle triplanar mapping and registry lookup
                        let normal = Self::calculate_local_normal_fast(chunk, x, y, z);
                        let abs_x = normal.x.abs();
                        let abs_y = normal.y.abs();
                        let abs_z = normal.z.abs();

                        let (u, v) = if abs_x > abs_y && abs_x > abs_z {
                            (
                                (world_base_z + z as i32).rem_euclid(256) as usize,
                                (world_base_y + y as i32).rem_euclid(256) as usize,
                            )
                        } else if abs_y > abs_z {
                            (
                                (world_base_x + x as i32).rem_euclid(256) as usize,
                                (world_base_z + z as i32).rem_euclid(256) as usize,
                            )
                        } else {
                            (
                                (world_base_x + x as i32).rem_euclid(256) as usize,
                                (world_base_y + y as i32).rem_euclid(256) as usize,
                            )
                        };

                        if let Some(baked_type) = dictionary.get(&(mat_id as u16)) {
                            let baked_gpu_payload = baked_type.texture[v * 256 + u];

                            // Inject high-quality smooth normals into the baked payload
                            let nx_4b = ((normal.x + 1.0) * 0.5 * 15.0).round() as u32;
                            let ny_4b = ((normal.y + 1.0) * 0.5 * 15.0).round() as u32;
                            let nz_4b = ((normal.z + 1.0) * 0.5 * 15.0).round() as u32;

                            final_gpu_val = (baked_gpu_payload & !(0xFFF << 15))
                                | ((nx_4b & 0xF) << 15)
                                | ((ny_4b & 0xF) << 19)
                                | ((nz_4b & 0xF) << 23);
                        } else {
                            final_gpu_val = 0;
                        }
                    } else {
                        // PAINT/SOLID PATH: Build GPU payload from raw CPU color/extras
                        let color_15b = cpu_packed & 0x7FFF;
                        let extras = (cpu_packed >> 24) & 0xF;

                        let normal = Self::calculate_local_normal_fast(chunk, x, y, z);
                        let nx_4b = ((normal.x + 1.0) * 0.5 * 15.0).round() as u32;
                        let ny_4b = ((normal.y + 1.0) * 0.5 * 15.0).round() as u32;
                        let nz_4b = ((normal.z + 1.0) * 0.5 * 15.0).round() as u32;

                        final_gpu_val = color_15b
                            | (nx_4b << 15)
                            | (ny_4b << 19)
                            | (nz_4b << 23)
                            | ((extras & 0xF) << 27);
                    }

                    // --- 7. ADD TO SPARSE ARRAY ---
                    // Push only the visible voxel to the flat vector
                    self.gpu_upload_data.sparse_visuals.push(final_gpu_val);
                    self.gpu_upload_data.voxel_count += 1;

                    // Clear lowest set bit to continue iteration
                    bits &= bits - 1;
                }
            }
        }
    }

    #[func]
    pub fn texture_chunk_32(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
        let terrain_manager = if let Some(tm) = &self.terrain_manager {
            tm.bind()
        } else {
            return;
        };

        let dictionary = &terrain_manager.baked_registry.voxel_dictionary;

        // 1. Get the chunk data
        let chunk = if let Some(data) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
            data
        } else {
            return;
        };

        // 2. Clear the GPU buffer for this update (ensure empty voxels are 0)
        self.gpu_buffer.fill(0);

        let world_base_x = cx * 64;
        let world_base_y = cy * 64;
        let world_base_z = cz * 64;

        for z in 0..64 {
            let z_mask_off = z * 64;
            let z_vis_off = z * 4096;

            let zm_off = if z > 0 { (z - 1) * 64 } else { z_mask_off };
            let zp_off = if z < 63 { (z + 1) * 64 } else { z_mask_off };
            let is_on_z_boundary = z == 0 || z == 63;

            for y in 0..64 {
                let mask = chunk.occupancy[z_mask_off + y];
                if mask == 0 {
                    continue;
                }

                let ym_mask = if y > 0 {
                    chunk.occupancy[z_mask_off + y - 1]
                } else {
                    mask
                };
                let yp_mask = if y < 63 {
                    chunk.occupancy[z_mask_off + y + 1]
                } else {
                    mask
                };
                let zm_mask = chunk.occupancy[zm_off + y];
                let zp_mask = chunk.occupancy[zp_off + y];

                // Exposure logic to skip internal hidden voxels
                let hidden_by_mirror =
                    (mask << 1) & (mask >> 1) & ym_mask & yp_mask & zm_mask & zp_mask;
                let is_on_x_boundary = mask & 0x8000000000000001;
                let is_on_y_boundary = y == 0 || y == 63;

                let mut exposed_mask = mask & !hidden_by_mirror;
                if is_on_y_boundary || is_on_z_boundary || (is_on_x_boundary != 0) {
                    let edge_bits = if is_on_y_boundary || is_on_z_boundary {
                        mask
                    } else {
                        is_on_x_boundary
                    };
                    exposed_mask |= (mask & edge_bits);
                }

                let mut bits = exposed_mask;
                while bits != 0 {
                    let x = bits.trailing_zeros() as usize;
                    let vis_idx = z_vis_off + (y * 64) + x;

                    // --- CPU DATA EXTRACTION ---
                    // [4-bit Empty][4-bit Extras][8-bit Material][1-bit Tex][15-bit Color]
                    let cpu_packed = chunk.visuals[vis_idx];
                    let mat_id = (cpu_packed >> 16) & 0xFF;
                    let has_tex = (cpu_packed >> 15) & 1 == 1;

                    // --- GPU PAYLOAD PREPARATION ---
                    let mut final_gpu_val: u32;

                    if has_tex {
                        // --- TEXTURED PATH: Sample Baked Registry ---
                        // Calculate Smooth Normal for the GPU
                        let normal = Self::calculate_local_normal_fast(chunk, x, y, z);

                        // Triplanar UV mapping
                        let abs_x = normal.x.abs();
                        let abs_y = normal.y.abs();
                        let abs_z = normal.z.abs();

                        let (u, v) = if abs_x > abs_y && abs_x > abs_z {
                            (
                                (world_base_z + z as i32).rem_euclid(256) as usize,
                                (world_base_y + y as i32).rem_euclid(256) as usize,
                            )
                        } else if abs_y > abs_z {
                            (
                                (world_base_x + x as i32).rem_euclid(256) as usize,
                                (world_base_z + z as i32).rem_euclid(256) as usize,
                            )
                        } else {
                            (
                                (world_base_x + x as i32).rem_euclid(256) as usize,
                                (world_base_y + y as i32).rem_euclid(256) as usize,
                            )
                        };

                        if let Some(baked_type) = dictionary.get(&(mat_id as u16)) {
                            let baked_gpu_payload = baked_type.texture[v * 256 + u];

                            // Overwrite the baked neutral normal with our high-quality smooth normal
                            let nx_4b = ((normal.x + 1.0) * 0.5 * 15.0).round() as u32;
                            let ny_4b = ((normal.y + 1.0) * 0.5 * 15.0).round() as u32;
                            let nz_4b = ((normal.z + 1.0) * 0.5 * 15.0).round() as u32;

                            // Strip baked normal (bits 15-26) and inject smooth normal
                            final_gpu_val = (baked_gpu_payload & !(0xFFF << 15))
                                | ((nx_4b & 0xF) << 15)
                                | ((ny_4b & 0xF) << 19)
                                | ((nz_4b & 0xF) << 23);
                        } else {
                            final_gpu_val = 0; // Fallback
                        }
                    } else {
                        // --- PAINT/SOLID PATH: Build from CPU data ---
                        let color_15b = cpu_packed & 0x7FFF;
                        let extras = (cpu_packed >> 24) & 0xF; // emit, metal, trans, rough

                        let normal = Self::calculate_local_normal_fast(chunk, x, y, z);
                        let nx_4b = ((normal.x + 1.0) * 0.5 * 15.0).round() as u32;
                        let ny_4b = ((normal.y + 1.0) * 0.5 * 15.0).round() as u32;
                        let nz_4b = ((normal.z + 1.0) * 0.5 * 15.0).round() as u32;

                        final_gpu_val = color_15b
                            | (nx_4b << 15)
                            | (ny_4b << 19)
                            | (nz_4b << 23)
                            | ((extras & 0xF) << 27);
                    }

                    // Write to the Terrain Node's shared GPU buffer
                    self.gpu_buffer[vis_idx] = final_gpu_val;

                    bits &= bits - 1;
                }
            }
        }
    }

    // #[func]
    // pub fn texture_chunk_mirror(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
    //     let terrain_manager = if let Some(tm) = &self.terrain_manager {
    //         tm.bind()
    //     } else {
    //         return;
    //     };
    //     let dictionary = &terrain_manager.baked_registry.voxel_dictionary;
    //     let chunk = if let Some(data) = self.octree.get_node_data_mut(cx, cy, cz, depth) {
    //         data
    //     } else {
    //         return;
    //     };

    //     let world_base_x = cx * 64;
    //     let world_base_y = cy * 64;
    //     let world_base_z = cz * 64;

    //     for z in 0..64 {
    //         let z_mask_off = z * 64;
    //         let z_vis_off = z * 4096;

    //         // Pre-cache Z-slice offsets with Mirroring for borders
    //         let zm_off = if z > 0 { (z - 1) * 64 } else { z_mask_off };
    //         let zp_off = if z < 63 { (z + 1) * 64 } else { z_mask_off };

    //         let is_on_z_boundary = z == 0 || z == 63;

    //         for y in 0..64 {
    //             let mask = chunk.occupancy[z_mask_off + y];
    //             if mask == 0 {
    //                 continue;
    //             }

    //             // Cache Y-neighbor masks with Mirroring
    //             let ym_mask = if y > 0 {
    //                 chunk.occupancy[z_mask_off + y - 1]
    //             } else {
    //                 mask
    //             };
    //             let yp_mask = if y < 63 {
    //                 chunk.occupancy[z_mask_off + y + 1]
    //             } else {
    //                 mask
    //             };

    //             // Cache Z-neighbor masks
    //             let zm_mask = chunk.occupancy[zm_off + y];
    //             let zp_mask = chunk.occupancy[zp_off + y];

    //             // 1. Calculate the standard Mirrored Hidden Mask
    //             // This returns 1 for bits that are completely surrounded by solid neighbors (clamped/mirrored)
    //             let hidden_by_mirror =
    //                 (mask << 1) & (mask >> 1) & ym_mask & yp_mask & zm_mask & zp_mask;

    //             // 2. Identify Boundary Bits
    //             let is_on_x_boundary = mask & 0x8000000000000001; // Bit 0 and Bit 63
    //             let is_on_y_boundary = y == 0 || y == 63;

    //             // 3. The "Edge-Aware" Exposure Mask
    //             // A voxel is exposed if:
    //             // (It's NOT hidden by mirroring) OR (It's on a physical chunk boundary AND it is solid)
    //             let mut exposed_mask = mask & !hidden_by_mirror;

    //             if is_on_y_boundary || is_on_z_boundary || (is_on_x_boundary != 0) {
    //                 let mut edge_bits = 0u64;
    //                 if is_on_y_boundary || is_on_z_boundary {
    //                     edge_bits = mask;
    //                 } else {
    //                     edge_bits = is_on_x_boundary;
    //                 }
    //                 // Force exposure for solid boundary voxels that were "hidden" by the mirror clamp
    //                 exposed_mask |= (mask & edge_bits);
    //             }

    //             if exposed_mask == 0 {
    //                 continue;
    //             }

    //             let mut bits = exposed_mask;
    //             while bits != 0 {
    //                 let x = bits.trailing_zeros() as usize;
    //                 let vis_idx = z_vis_off + (y * 64) + x;
    //                 let packed_val = chunk.visuals[vis_idx];

    //                 // 4. Get Normal and Triplanar UVs
    //                 let normal = Self::calculate_local_normal_fast(chunk, x, y, z);

    //                 let abs_x = normal.x.abs();
    //                 let abs_y = normal.y.abs();
    //                 let abs_z = normal.z.abs();

    //                 let (u, v) = if abs_x > abs_y && abs_x > abs_z {
    //                     (
    //                         (world_base_z + z as i32).rem_euclid(256) as usize,
    //                         (world_base_y + y as i32).rem_euclid(256) as usize,
    //                     )
    //                 } else if abs_y > abs_z {
    //                     (
    //                         (world_base_x + x as i32).rem_euclid(256) as usize,
    //                         (world_base_z + z as i32).rem_euclid(256) as usize,
    //                     )
    //                 } else {
    //                     (
    //                         (world_base_x + x as i32).rem_euclid(256) as usize,
    //                         (world_base_y + y as i32).rem_euclid(256) as usize,
    //                     )
    //                 };

    //                 // 5. Look up Baked Texture Data
    //                 let mat_id = (packed_val >> 16) & 0xFF;
    //                 if let Some(baked_type) = dictionary.get(&(mat_id as u16)) {
    //                     let baked_visual = baked_type.texture[v * 256 + u];

    //                     // 6. Update Visuals (Preserving high metadata bits)
    //                     chunk.visuals[vis_idx] =
    //                         (packed_val & 0xFF000000) | (mat_id << 16) | (baked_visual as u32);
    //                 }

    //                 bits &= bits - 1; // Clear lowest set bit
    //             }
    //         }
    //     }
    //     chunk.is_dirty = true;
    // }

    // #[func]
    // pub fn texture_chunk(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
    //     let terrain_manager = if let Some(tm) = &self.terrain_manager {
    //         tm.bind()
    //     } else {
    //         return;
    //     };
    //     let dictionary = &terrain_manager.baked_registry.voxel_dictionary;

    //     let chunk = if let Some(data) = self.octree.get_leaf_data_mut(cx, cy, cz) {
    //         data
    //     } else {
    //         return;
    //     };

    //     let world_base_x = cx * 64;
    //     let world_base_y = cy * 64;
    //     let world_base_z = cz * 64;

    //     for z in 0..64 {
    //         let z_mask_off = z * 64;
    //         let z_vis_off = z * 4096;

    //         // Pre-cache Z-slice offsets with Mirroring for borders
    //         let zm_off = if z > 0 { (z - 1) * 64 } else { z_mask_off };
    //         let zp_off = if z < 63 { (z + 1) * 64 } else { z_mask_off };

    //         for y in 0..64 {
    //             let mask = chunk.occupancy[z_mask_off + y];
    //             if mask == 0 {
    //                 continue;
    //             }

    //             // Cache Y-neighbor masks with Mirroring
    //             let ym_mask = if y > 0 {
    //                 chunk.occupancy[z_mask_off + y - 1]
    //             } else {
    //                 mask
    //             };
    //             let yp_mask = if y < 63 {
    //                 chunk.occupancy[z_mask_off + y + 1]
    //             } else {
    //                 mask
    //             };

    //             // Cache Z-neighbor masks
    //             let zm_mask = chunk.occupancy[zm_off + y];
    //             let zp_mask = chunk.occupancy[zp_off + y];

    //             // 1. SIMD-style Exposure Mask
    //             // A voxel is exposed if it's occupied AND at least one 6-way neighbor is air
    //             let exposed_mask =
    //                 mask & !((mask << 1) & (mask >> 1) & ym_mask & yp_mask & zm_mask & zp_mask);

    //             if exposed_mask == 0 {
    //                 continue;
    //             }

    //             // 2. Process only the bits that are actually exposed
    //             let mut bits = exposed_mask;
    //             while bits != 0 {
    //                 let x = bits.trailing_zeros() as usize;
    //                 let vis_idx = z_vis_off + (y * 64) + x;

    //                 let current_visual = chunk.visuals[vis_idx];
    //                 let mat_id = ((current_visual >> 16) & 0xFF) as u16;

    //                 if let Some(baked_type) = dictionary.get(&mat_id) {
    //                     // 3. Fast Normal Estimation
    //                     let normal = Self::calculate_local_normal_fast(chunk, x, y, z);

    //                     // 4. Triplanar Axis Selection & World UV calculation
    //                     let abs_x = normal.x.abs();
    //                     let abs_y = normal.y.abs();
    //                     let abs_z = normal.z.abs();

    //                     let (u, v) = if abs_x > abs_y && abs_x > abs_z {
    //                         // X is dominant: project on ZY plane
    //                         (
    //                             (world_base_z + z as i32).rem_euclid(256) as usize,
    //                             (world_base_y + y as i32).rem_euclid(256) as usize,
    //                         )
    //                     } else if abs_y > abs_z {
    //                         // Y is dominant: project on XZ plane
    //                         (
    //                             (world_base_x + x as i32).rem_euclid(256) as usize,
    //                             (world_base_z + z as i32).rem_euclid(256) as usize,
    //                         )
    //                     } else {
    //                         // Z is dominant: project on XY plane
    //                         (
    //                             (world_base_x + x as i32).rem_euclid(256) as usize,
    //                             (world_base_y + y as i32).rem_euclid(256) as usize,
    //                         )
    //                     };

    //                     // 5. Read pre-packed u16 visual data (RRRR GGGG BBBB EEMT)
    //                     let baked_visual = baked_type.texture[v * 256 + u];

    //                     // 6. Re-pack: Preserve Metadata (24-31) | MatID (16-23) | Baked Colors (0-15)
    //                     chunk.visuals[vis_idx] = (current_visual & 0xFF000000)
    //                         | ((mat_id as u32) << 16)
    //                         | (baked_visual as u32);
    //                 }

    //                 // 3. Fast Normal Estimation with Mirroring
    //                 // let normal = Self::calculate_local_normal_fast(chunk, x, y, z);

    //                 // // Pack directly into the visuals array
    //                 // let r = ((normal.x + 1.0) * 7.5) as u32;
    //                 // let g = ((normal.y + 1.0) * 7.5) as u32;
    //                 // let b = ((normal.z + 1.0) * 7.5) as u32;

    //                 // chunk.visuals[z_vis_off + (y * 64) + x] =
    //                 //     Self::pack_visuals(0, r, g, b, 0, false, false);

    //                 bits &= bits - 1; // Clear the lowest set bit
    //             }
    //         }
    //     }
    //     chunk.is_dirty = true;
    // }

    // #[func]
    // pub fn color_chunk_by_normals(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) {
    //     let chunk = if let Some(data) = self.octree.get_leaf_data_mut(cx, cy, cz) {
    //         data
    //     } else {
    //         return;
    //     };

    //     for z in 0..64 {
    //         let z_mask_off = z * 64;
    //         let z_vis_off = z * 4096;

    //         // Pre-cache Z-slice offsets with Mirroring for borders
    //         let zm_off = if z > 0 { (z - 1) * 64 } else { z_mask_off };
    //         let zp_off = if z < 63 { (z + 1) * 64 } else { z_mask_off };

    //         for y in 0..64 {
    //             let mask = chunk.occupancy[z_mask_off + y];
    //             if mask == 0 {
    //                 continue;
    //             }

    //             // Cache Y-neighbor masks with Mirroring
    //             let ym_mask = if y > 0 {
    //                 chunk.occupancy[z_mask_off + y - 1]
    //             } else {
    //                 mask
    //             };
    //             let yp_mask = if y < 63 {
    //                 chunk.occupancy[z_mask_off + y + 1]
    //             } else {
    //                 mask
    //             };

    //             // Cache Z-neighbor masks
    //             let zm_mask = chunk.occupancy[zm_off + y];
    //             let zp_mask = chunk.occupancy[zp_off + y];

    //             // 1. SIMD-style Exposure Mask
    //             // A voxel is exposed if it's occupied AND at least one 6-way neighbor is air
    //             let exposed_mask =
    //                 mask & !((mask << 1) & (mask >> 1) & ym_mask & yp_mask & zm_mask & zp_mask);

    //             if exposed_mask == 0 {
    //                 continue;
    //             }

    //             // 2. Process only the bits that are actually exposed
    //             let mut bits = exposed_mask;
    //             while bits != 0 {
    //                 let x = bits.trailing_zeros() as usize;

    //                 // 3. Fast Normal Estimation with Mirroring
    //                 let normal = Self::calculate_local_normal_fast(chunk, x, y, z);

    //                 // Pack directly into the visuals array
    //                 let r = ((normal.x + 1.0) * 7.5) as u32;
    //                 let g = ((normal.y + 1.0) * 7.5) as u32;
    //                 let b = ((normal.z + 1.0) * 7.5) as u32;

    //                 chunk.visuals[z_vis_off + (y * 64) + x] =
    //                     Self::pack_visuals(0, r, g, b, false, false, false, true);

    //                 bits &= bits - 1; // Clear the lowest set bit
    //             }
    //         }
    //     }
    //     chunk.is_dirty = true;
    // }

    #[inline(always)]
    fn calculate_local_normal_fast(chunk: &ChunkData, x: usize, y: usize, z: usize) -> Vector3 {
        let mut nx = 0.0;
        let mut ny = 0.0;
        let mut nz = 0.0;

        // We sample a 3x3x3 area. To keep it fast, we use a single loop or flattened logic.
        for dz in -1..=1 {
            let z_idx = ((z as i32 + dz).clamp(0, 63) as usize) * 64;
            for dy in -1..=1 {
                let y_idx = (y as i32 + dy).clamp(0, 63) as usize;
                let mask = chunk.occupancy[z_idx + y_idx];

                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }

                    // Check neighbor bit with Mirroring
                    let nx_bit = (x as i32 + dx).clamp(0, 63) as usize;
                    if (mask & (1u64 << nx_bit)) != 0 {
                        nx -= dx as f32;
                        ny -= dy as f32;
                        nz -= dz as f32;
                    }
                }
            }
        }

        let length_sq = nx * nx + ny * ny + nz * nz;
        // If len_sq is effectively zero, we are either:
        // 1. Fully surrounded by solid voxels (underground)
        // 2. At a chunk border where clamping made the vector cancel out
        if length_sq == 0.0 {
            // Return a clean Y-Up vector to Godot
            return Vector3::new(0.0, 1.0, 0.0);
        }

        if length_sq > 0.01 {
            let inv_len = 1.0 / length_sq.sqrt();
            Vector3::new(nx * inv_len, ny * inv_len, nz * inv_len)
        } else {
            Vector3::UP
        }
    }

    // #[func]
    // pub fn place_vox_model(&mut self, path: GString, world_origin: Vector3i) {
    //     // 1. Convert Godot "res://" path to a global OS path
    //     let global_path = ProjectSettings::singleton().globalize_path(&path.clone());
    //     let path_str = global_path.to_string();

    //     // 2. Load and parse the .vox file
    //     let mut file = match File::open(&path_str) {
    //         Ok(f) => f,
    //         Err(e) => {
    //             godot_error!("Failed to open .vox file at {}: {}", path_str, e);
    //             return;
    //         }
    //     };

    //     let mut buffer = Vec::new();
    //     file.read_to_end(&mut buffer).unwrap();

    //     let data = match dot_vox::load_bytes(&buffer) {
    //         Ok(d) => d,
    //         Err(e) => {
    //             godot_error!("Failed to parse .vox data: {}", e);
    //             return;
    //         }
    //     };

    //     // Track affected chunks to call update_texture_32 and mesh() later
    //     let mut affected_chunks = HashMap::new();
    //     let chunk_size = 64;

    //     // 3. Iterate through all models in the file
    //     for model in data.models.iter() {
    //         for voxel in model.voxels.iter() {
    //             // MagicaVoxel palette lookup
    //             let rgba = data.palette[voxel.i as usize];

    //             // Convert RGBA8 to 5-bit channels
    //             // let r = (rgba.r >> 3) as u32;
    //             // let g = (rgba.g >> 3) as u32;
    //             // let b = (rgba.b >> 3) as u32;
    //             let r = rgba.r as u32;
    //             let g = rgba.g as u32;
    //             let b = rgba.b as u32;

    //             // Calculate absolute world position (Swizzle Z and Y for Godot orientation)
    //             let voxel_pos =
    //                 world_origin + Vector3i::new(voxel.x as i32, voxel.z as i32, voxel.y as i32);

    //             // 4. Place voxel using high-level logic
    //             self.set_voxel_full(
    //                 voxel_pos.x,
    //                 voxel_pos.y,
    //                 voxel_pos.z,
    //                 0,
    //                 r,
    //                 g,
    //                 b,
    //                 false,
    //                 false,
    //                 false,
    //                 true,
    //                 true,
    //                 false,
    //                 false,
    //             );

    //             // 5. Calculate which chunk this voxel belongs to
    //             // Using Euclidean division to handle negative coordinates correctly
    //             let cx = (voxel_pos.x as f32 / chunk_size as f32).floor() as i32;
    //             let cy = (voxel_pos.y as f32 / chunk_size as f32).floor() as i32;
    //             let cz = (voxel_pos.z as f32 / chunk_size as f32).floor() as i32;

    //             let chunk_coord = Vector3i::new(cx, cy, cz);

    //             affected_chunks.insert(chunk_coord, true);
    //         }
    //     }

    //     // 6. Finalize: Update GPU textures and regenerate meshes for all touched chunks
    //     for (coord, _) in affected_chunks {
    //         self.texture_chunk_32(coord.x, coord.y, coord.z, 0);
    //         self.update_chunk_texture_32(coord.x, coord.y, coord.z, 0);
    //         self.mesh_chunk(coord.x, coord.y, coord.z, false, 0, true);
    //     }

    //     godot_print!("Successfully placed .vox model and updated chunks.");
    // }

    #[func]
    pub fn place_vox_model(&mut self, path: GString, world_origin: Vector3i, use_occlusion: bool) {
        let global_path = ProjectSettings::singleton().globalize_path(&path.clone());
        let mut file = File::open(global_path.to_string()).expect("Failed to open .vox");
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let data = dot_vox::load_bytes(&buffer).expect("Failed to parse .vox");
        if data.scenes.is_empty() {
            return;
        }

        let mut affected_chunks: HashMap<Vector3i, bool> = HashMap::new();

        // Start recursion with an Identity transform (No rotation, No translation)
        self.process_node(
            &data,
            0,
            Transform3D::IDENTITY,
            world_origin,
            &mut affected_chunks,
        );

        // Update Chunks logic...
        for (coord, _) in affected_chunks {
            self.texture_chunk_sparse(coord.x, coord.y, coord.z, 0);
            self.update_chunk_texture_sparse(coord.x, coord.y, coord.z, 0);
            self.mesh_chunk(coord.x, coord.y, coord.z, false, 0, true, use_occlusion);
        }
    }

    fn process_node(
        &mut self,
        data: &dot_vox::DotVoxData,
        node_idx: usize,
        accumulated_transform: Transform3D,
        world_origin: Vector3i,
        affected_chunks: &mut HashMap<Vector3i, bool>,
    ) {
        let node = &data.scenes[node_idx];

        match node {
            dot_vox::SceneNode::Transform {
                attributes,
                frames,
                child,
                ..
            } => {
                // Attributes can be in the node or the first frame depending on MV version
                let frame_attrs = frames.first().map(|f| &f.attributes);

                // 1. Parse Translation (_t)
                let mut local_t = Vector3::ZERO;
                if let Some(t_str) = attributes
                    .get("_t")
                    .or(frame_attrs.and_then(|a| a.get("_t")))
                {
                    let parts: Vec<&str> = t_str.split_whitespace().collect();
                    if parts.len() == 3 {
                        local_t.x = parts[0].parse::<f32>().unwrap_or(0.0);
                        local_t.y = parts[1].parse::<f32>().unwrap_or(0.0);
                        local_t.z = parts[2].parse::<f32>().unwrap_or(0.0);
                    }
                }

                // 2. Parse Rotation (_r)
                let mut local_r = Basis::IDENTITY;
                if let Some(r_str) = attributes
                    .get("_r")
                    .or(frame_attrs.and_then(|a| a.get("_r")))
                {
                    local_r = Self::parse_vox_rotation(r_str);
                }

                // 3. Accumulate: New Transform = Parent * Local
                let local_transform = Transform3D::new(local_r, local_t);
                let next_transform = accumulated_transform * local_transform;

                self.process_node(
                    data,
                    *child as usize,
                    next_transform,
                    world_origin,
                    affected_chunks,
                );
            }

            dot_vox::SceneNode::Group { children, .. } => {
                for &child_idx in children {
                    self.process_node(
                        data,
                        child_idx as usize,
                        accumulated_transform,
                        world_origin,
                        affected_chunks,
                    );
                }
            }

            dot_vox::SceneNode::Shape { models, .. } => {
                for model_instance in models {
                    let model = &data.models[model_instance.model_id as usize];

                    // MagicaVoxel centers models; subtract half-size to get the pivot
                    let size_offset = Vector3::new(
                        (model.size.x as f32 / 2.0).floor(),
                        (model.size.y as f32 / 2.0).floor(),
                        (model.size.z as f32 / 2.0).floor(),
                    );

                    for voxel in &model.voxels {
                        let rgba = data.palette[voxel.i as usize];
                        let v_local = Vector3::new(voxel.x as f32, voxel.y as f32, voxel.z as f32);

                        // 1. Position in MagicaVoxel World Space (Centered and Transformed)
                        let p_vox_world = accumulated_transform * (v_local - size_offset);

                        // 2. Swizzle to Godot Space (Vox X,Y,Z -> Godot X,Z,Y)
                        // Note: We use your existing (X, Z, Y) mapping
                        let godot_pos = Vector3i::new(
                            p_vox_world.x.round() as i32,
                            p_vox_world.z.round() as i32,
                            p_vox_world.y.round() as i32,
                        ) + world_origin;

                        self.set_voxel_full(
                            godot_pos.x,
                            godot_pos.y,
                            godot_pos.z,
                            0,
                            rgba.r as u32,
                            rgba.g as u32,
                            rgba.b as u32,
                            false,
                            false,
                            false,
                            true,
                            true,
                            false,
                            false,
                        );

                        let cx = godot_pos.x.div_euclid(64);
                        let cy = godot_pos.y.div_euclid(64);
                        let cz = godot_pos.z.div_euclid(64);
                        affected_chunks.insert(Vector3i::new(cx, cy, cz), true);
                    }
                }
            }
        }
    }

    fn parse_vox_rotation(r_str: &str) -> Basis {
        let r = r_str.parse::<u8>().unwrap_or(0);

        let i0 = (r & 3) as usize;
        let i1 = ((r >> 2) & 3) as usize;
        let i2 = [0, 1, 2]
            .iter()
            .find(|&&x| x != i0 && x != i1)
            .cloned()
            .unwrap_or(2);

        let s0 = if (r >> 4) & 1 == 1 { -1.0 } else { 1.0 };
        let s1 = if (r >> 5) & 1 == 1 { -1.0 } else { 1.0 };
        let s2 = if (r >> 6) & 1 == 1 { -1.0 } else { 1.0 };

        // Create three empty columns
        let mut cols = [Vector3::ZERO, Vector3::ZERO, Vector3::ZERO];

        // MagicaVoxel _r defines which component is non-zero for each ROW.
        // Row 0 (X) has its non-zero at column i0 with sign s0
        cols[i0][Vector3Axis::X] = s0;
        // Row 1 (Y) has its non-zero at column i1 with sign s1
        cols[i1][Vector3Axis::Y] = s1;
        // Row 2 (Z) has its non-zero at column i2 with sign s2
        cols[i2][Vector3Axis::Z] = s2;

        Basis::from_cols(cols[0], cols[1], cols[2])
    }

    #[func]
    pub fn physics_check_big(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) {
        let _clumps = self.extract_floating_clumps_256(Vector3i {
            x: (cx),
            y: (cy),
            z: (cz),
        });
    }

    #[func]
    pub fn physics_check_medium(&mut self, cx: i32, cy: i32, cz: i32, _depth: i32) {
        let clumps = self.extract_floating_clumps_128(Vector3i {
            x: cx,
            y: cy,
            z: cz,
        });

        if clumps.is_empty() {
            return;
        }

        // Get a reference to the parent (usually the terrain node itself or a "Debris" container)
        let mut parent = self
            .base()
            .get_parent()
            .expect("Terrain node must be in scene tree");

        for clump in clumps {
            // 1. Instantiate the VoxelRigidBody
            let mut new_body = VoxelRigidBody::new_alloc();

            // 2. Pass the mesh resource from the terrain manager (assuming you have it stored)
            // You might need to store a reference to the voxel box mesh in your TerrainManager

            // 3. Initialize with the 128 (Large) tier data
            new_body.bind_mut().init_clump(64, clump);

            // 4. Add to scene tree
            parent.add_child(new_body.to_godot());
        }
    }

    // pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
    //     let mut affected_chunks = HashMap::new();

    //     // 3x3x3 chunks, each 64 voxels → 192 voxels → 96 coarse cells at 1/2 res
    //     const CN: usize = 3; // chunks per axis
    //     const N: usize = 96; // coarse grid size (CN * 64 / 2)
    //     const N2: usize = N * N;
    //     const N3: usize = N * N * N;

    //     let mut coarse_grid = vec![0u8; N3];

    //     // --- PASS 1: COLLECT OCCUPANCY ---
    //     for lz in 0..CN as i32 {
    //         for ly in 0..CN as i32 {
    //             for lx in 0..CN as i32 {
    //                 let gx_base = (lx * 32) as usize;
    //                 let gy_base = (ly * 32) as usize;
    //                 let gz_base = (lz * 32) as usize;
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_offset = gz * N2;
    //                         let mut y = 0;
    //                         while y + 4 <= 64 {
    //                             let cols = u64x4::new([
    //                                 chunk.occupancy[z * 64 + y],
    //                                 chunk.occupancy[z * 64 + y + 1],
    //                                 chunk.occupancy[z * 64 + y + 2],
    //                                 chunk.occupancy[z * 64 + y + 3],
    //                             ]);
    //                             if cols == u64x4::ZERO {
    //                                 y += 4;
    //                                 continue;
    //                             }
    //                             for dy in 0..4 {
    //                                 let col = chunk.occupancy[z * 64 + y + dy];
    //                                 if col == 0 {
    //                                     continue;
    //                                 }
    //                                 let gy = gy_base + ((y + dy) >> 1);
    //                                 let gy_offset = gz_offset + gy * N;
    //                                 let mut bits = col;
    //                                 while bits != 0 {
    //                                     let x = bits.trailing_zeros() as usize;
    //                                     coarse_grid[gy_offset + gx_base + (x >> 1)] = 1;
    //                                     bits &= bits - 1;
    //                                     bits &= !(1u64 << (x ^ 1));
    //                                 }
    //                             }
    //                             y += 4;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- EARLY EXIT ---
    //     let border = N - 1;
    //     let has_interior_solid = {
    //         let mut found = false;
    //         'outer: for z in 1..border {
    //             let gz_off = z * N2;
    //             for y in 1..border {
    //                 let row_start = gz_off + y * N + 1;
    //                 let row = &coarse_grid[row_start..row_start + border - 1];
    //                 let mut i = 0;
    //                 while i + 8 <= row.len() {
    //                     let word = u64::from_ne_bytes(row[i..i + 8].try_into().unwrap());
    //                     if word != 0 {
    //                         found = true;
    //                         break 'outer;
    //                     }
    //                     i += 8;
    //                 }
    //                 while i < row.len() {
    //                     if row[i] != 0 {
    //                         found = true;
    //                         break 'outer;
    //                     }
    //                     i += 1;
    //                 }
    //             }
    //         }
    //         found
    //     };

    //     if !has_interior_solid {
    //         return Vec::new();
    //     }

    //     // --- PASS 2: SEED + FLOOD GROUNDED ---
    //     let mut stack: Vec<u32> = Vec::with_capacity(8192);
    //     macro_rules! try_seed {
    //         ($idx:expr) => {
    //             if coarse_grid[$idx] == 1 {
    //                 coarse_grid[$idx] = 2;
    //                 stack.push($idx as u32);
    //             }
    //         };
    //     }
    //     for y in 0..N {
    //         for x in 0..N {
    //             try_seed!(x + y * N);
    //             try_seed!(x + y * N + border * N2);
    //         }
    //     }
    //     for z in 1..border {
    //         for x in 0..N {
    //             try_seed!(x + z * N2);
    //             try_seed!(x + border * N + z * N2);
    //         }
    //     }
    //     for z in 1..border {
    //         for y in 1..border {
    //             try_seed!(y * N + z * N2);
    //             try_seed!(border + y * N + z * N2);
    //         }
    //     }
    //     self.flood_fill_coarse(&mut coarse_grid, &mut stack, 2, N);

    //     // --- PASS 3: LABEL CLUMPS ---
    //     let mut next_id = 10u8;
    //     for i in 0..N3 {
    //         if coarse_grid[i] == 1 {
    //             coarse_grid[i] = next_id;
    //             stack.push(i as u32);
    //             self.flood_fill_coarse(&mut coarse_grid, &mut stack, next_id, N);
    //             next_id = next_id.saturating_add(1);
    //             if next_id == 255 {
    //                 break;
    //             }
    //         }
    //     }

    //     let num_clumps = (next_id - 10) as usize;
    //     if num_clumps == 0 {
    //         return Vec::new();
    //     }

    //     // --- PASS 3.5: COUNT VOXELS PER CLUMP ---
    //     // We need counts before extraction so we can pre-allocate.
    //     // Do a fast coarse-grid scan: for each coarse cell with label>=10,
    //     // count how many fine voxels map to it. At 1/2 res each coarse cell
    //     // covers 2x2x2=8 fine voxels max, but actual count comes from occupancy.
    //     // Simpler: do one pre-scan pass over chunk occupancy just counting,
    //     // then a second pass extracting into pre-allocated slices.
    //     let mut voxel_counts = vec![0u32; num_clumps];

    //     for lz in 0..CN as i32 {
    //         for ly in 0..CN as i32 {
    //             for lx in 0..CN as i32 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let gx_base = (lx * 32) as usize;
    //                     let gy_base = (ly * 32) as usize;
    //                     let gz_base = (lz * 32) as usize;
    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_off = gz * N2;
    //                         for y in 0..64usize {
    //                             let col = chunk.occupancy[z * 64 + y];
    //                             if col == 0 {
    //                                 continue;
    //                             }
    //                             let gy = gy_base + (y >> 1);
    //                             let gy_off = gz_off + gy * N;
    //                             let mut bits = col;
    //                             while bits != 0 {
    //                                 let x = bits.trailing_zeros() as usize;
    //                                 bits &= bits - 1;
    //                                 let gx = gx_base + (x >> 1);
    //                                 let label = coarse_grid[gy_off + gx];
    //                                 if label >= 10 {
    //                                     voxel_counts[(label - 10) as usize] += 1;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- Pre-allocate flat staging buffer ---
    //     // All voxels for all clumps live in one contiguous allocation.
    //     // clump_offsets[i] = start index in staging buffer for clump i.
    //     let total_voxels: usize = voxel_counts.iter().map(|&c| c as usize).sum();
    //     let mut staging: Vec<FloatingVoxel> = Vec::with_capacity(total_voxels);
    //     // Safety: we'll write exactly total_voxels entries; use set_len after filling.
    //     // Actually just push — capacity is pre-reserved so no reallocations occur.

    //     let mut clump_offsets = vec![0usize; num_clumps + 1];
    //     for i in 0..num_clumps {
    //         clump_offsets[i + 1] = clump_offsets[i] + voxel_counts[i] as usize;
    //     }
    //     // Write cursors per clump
    //     let mut write_pos = clump_offsets[..num_clumps].to_vec();

    //     // Resize staging to full size so we can index directly
    //     staging.resize(
    //         total_voxels,
    //         FloatingVoxel {
    //             local_pos: Vector3i::ZERO,
    //             color_data: 0,
    //         },
    //     );

    //     let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

    //     // --- PASS 4: EXTRACT into pre-allocated staging ---
    //     for lz in 0..CN as i32 {
    //         for ly in 0..CN as i32 {
    //             for lx in 0..CN as i32 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let mut modified = false;
    //                     let gx_base = (lx * 32) as usize;
    //                     let gy_base = (ly * 32) as usize;
    //                     let gz_base = (lz * 32) as usize;
    //                     let wx_base = ((base_pos.x + lx) * 64) as i32;
    //                     let wy_base = ((base_pos.y + ly) * 64) as i32;
    //                     let wz_base = ((base_pos.z + lz) * 64) as i32;

    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_off = gz * N2;
    //                         let wz = wz_base + z as i32;
    //                         for y in 0..64usize {
    //                             let col_idx = z * 64 + y;
    //                             let col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }
    //                             let gy = gy_base + (y >> 1);
    //                             let gy_off = gz_off + gy * N;
    //                             let wy = wy_base + y as i32;
    //                             let visual_row = z * 4096 + y * 64;
    //                             let mut col_new = col;
    //                             let mut col_remaining = col;
    //                             while col_remaining != 0 {
    //                                 let x = col_remaining.trailing_zeros() as usize;
    //                                 col_remaining &= col_remaining - 1;
    //                                 let gx = gx_base + (x >> 1);
    //                                 let label = coarse_grid[gy_off + gx];
    //                                 if label >= 10 {
    //                                     let id = (label - 10) as usize;
    //                                     let wx = wx_base + x as i32;
    //                                     let mp = &mut min_points[id];
    //                                     mp.x = mp.x.min(wx);
    //                                     mp.y = mp.y.min(wy);
    //                                     mp.z = mp.z.min(wz);
    //                                     let visual_idx = visual_row + x;
    //                                     let pos = write_pos[id];
    //                                     staging[pos] = FloatingVoxel {
    //                                         local_pos: Vector3i::new(wx, wy, wz),
    //                                         color_data: chunk.visuals[visual_idx],
    //                                     };
    //                                     write_pos[id] += 1;
    //                                     col_new &= !(1u64 << x);
    //                                     chunk.visuals[visual_idx] = 0;
    //                                     modified = true;
    //                                 }
    //                             }
    //                             chunk.occupancy[col_idx] = col_new;
    //                         }
    //                     }
    //                     if modified {
    //                         chunk.is_dirty = true;
    //                         let key =
    //                             Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz);
    //                         affected_chunks.insert(key, key);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- Slice staging buffer into per-clump results ---
    //     // One allocation, zero copies — just hand out owned Vecs by draining slices.
    //     let mut clump_results: Vec<FloatingClump> = Vec::with_capacity(num_clumps);
    //     {
    //         // Consume staging into per-clump vecs without copying.
    //         // We do this by splitting the staging vec progressively.
    //         let mut rest = staging;
    //         for i in 0..num_clumps {
    //             let count = voxel_counts[i] as usize;
    //             let m = min_points[i];

    //             // Split off the tail, keep the head as this clump's voxels
    //             let tail = rest.split_off(count);
    //             let mut voxels = rest;
    //             rest = tail;

    //             // Relocalize positions
    //             for v in &mut voxels {
    //                 v.local_pos -= m;
    //             }

    //             clump_results.push(FloatingClump {
    //                 world_origin: Vector3::new(m.x as f32, m.y as f32, m.z as f32),
    //                 voxels,
    //             });
    //         }
    //     }

    //     for cb in affected_chunks {
    //         let c = cb.0;
    //         self.texture_chunk_sparse(c.x, c.y, c.z, 0);
    //         self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
    //         self.mesh_chunk(c.x, c.y, c.z, false, 0, true);
    //     }

    //     clump_results
    // }

    // // Flood fill now takes N as a parameter since it varies
    // fn flood_fill_coarse(&self, grid: &mut [u8], stack: &mut Vec<u32>, fill_id: u8, n: usize) {
    //     let n = n as i32;
    //     let n2 = n * n;

    //     while let Some(idx) = stack.pop() {
    //         let idx = idx as i32;
    //         let x = idx % n;
    //         let y = (idx / n) % n;
    //         let z = idx / n2;

    //         macro_rules! check_neighbor {
    //             ($nx:expr, $ny:expr, $nz:expr) => {
    //                 let nx = $nx;
    //                 let ny = $ny;
    //                 let nz = $nz;
    //                 if nx >= 0 && nx < n && ny >= 0 && ny < n && nz >= 0 && nz < n {
    //                     let n_idx = (nz * n2 + ny * n + nx) as usize;
    //                     if grid[n_idx] == 1 {
    //                         grid[n_idx] = fill_id;
    //                         stack.push(n_idx as u32);
    //                     }
    //                 }
    //             };
    //         }

    //         check_neighbor!(x + 1, y, z);
    //         check_neighbor!(x - 1, y, z);
    //         check_neighbor!(x, y + 1, z);
    //         check_neighbor!(x, y - 1, z);
    //         check_neighbor!(x, y, z + 1);
    //         check_neighbor!(x, y, z - 1);
    //     }
    // }

    pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
        let mut affected_chunks = HashMap::new();

        let mut coarse_grid = vec![0u8; 64 * 64 * 64];
        const N: usize = 64;
        const N2: usize = N * N;

        for lz in 0..4i32 {
            for ly in 0..4i32 {
                for lx in 0..4i32 {
                    let gx_base = (lx * 16) as usize;
                    let gy_base = (ly * 16) as usize;
                    let gz_base = (lz * 16) as usize;
                    if let Some(chunk) = self.octree.get_node_data_mut(
                        base_pos.x + lx,
                        base_pos.y + ly,
                        base_pos.z + lz,
                        0,
                    ) {
                        for z in 0..64usize {
                            let gz = gz_base + (z >> 2);
                            let gz_offset = gz * N2;
                            let mut y = 0;
                            while y + 4 <= 64 {
                                let cols = u64x4::new([
                                    chunk.occupancy[z * 64 + y],
                                    chunk.occupancy[z * 64 + y + 1],
                                    chunk.occupancy[z * 64 + y + 2],
                                    chunk.occupancy[z * 64 + y + 3],
                                ]);
                                if cols == u64x4::ZERO {
                                    y += 4;
                                    continue;
                                }
                                for dy in 0..4 {
                                    let col = chunk.occupancy[z * 64 + y + dy];
                                    if col == 0 {
                                        continue;
                                    }
                                    let gy = gy_base + ((y + dy) >> 2);
                                    let gy_offset = gz_offset + gy * N;
                                    // Collapse 4 fine bits -> 1 coarse bit
                                    let mut bits = (col | (col >> 1) | (col >> 2) | (col >> 3))
                                        & 0x1111_1111_1111_1111u64;
                                    while bits != 0 {
                                        let fx = bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        let gx = gx_base + (fx >> 2);
                                        coarse_grid[gy_offset + gx] = 1;
                                    }
                                }
                                y += 4;
                            }
                        }
                    }
                }
            }
        }

        // Early exit: any interior solid? Interior = x/y/z in 1..63
        let has_interior_solid = {
            let mut found = false;
            'outer: for z in 1..63usize {
                let gz_off = z * N2;
                for y in 1..63usize {
                    let row_start = gz_off + y * N + 1; // x=1..62
                    let row = &coarse_grid[row_start..row_start + 62];
                    let mut i = 0;
                    while i + 8 <= row.len() {
                        let word = u64::from_ne_bytes(row[i..i + 8].try_into().unwrap());
                        if word != 0 {
                            found = true;
                            break 'outer;
                        }
                        i += 8;
                    }
                    while i < row.len() {
                        if row[i] != 0 {
                            found = true;
                            break 'outer;
                        }
                        i += 1;
                    }
                }
            }
            found
        };

        if !has_interior_solid {
            return Vec::new();
        }

        let mut stack: Vec<u32> = Vec::with_capacity(8192);
        {
            macro_rules! try_seed {
                ($idx:expr) => {
                    if coarse_grid[$idx] == 1 {
                        coarse_grid[$idx] = 2;
                        stack.push($idx as u32);
                    }
                };
            }
            for y in 0..N {
                for x in 0..N {
                    try_seed!(x + y * N);
                    try_seed!(x + y * N + 63 * N2);
                }
            }
            for z in 1..63usize {
                for x in 0..N {
                    try_seed!(x + z * N2);
                    try_seed!(x + 63 * N + z * N2);
                }
            }
            for z in 1..63usize {
                for y in 1..63usize {
                    try_seed!(y * N + z * N2);
                    try_seed!(63 + y * N + z * N2);
                }
            }
        }

        self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

        let mut next_id = 10u8;
        for i in 0..coarse_grid.len() {
            if coarse_grid[i] == 1 {
                coarse_grid[i] = next_id;
                stack.push(i as u32);
                self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
                next_id = next_id.saturating_add(1);
                if next_id == 255 {
                    break;
                }
            }
        }

        let num_clumps = (next_id - 10) as usize;
        if num_clumps == 0 {
            return Vec::new();
        }

        let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
            .map(|_| FloatingClump {
                world_origin: Vector3::ZERO,
                voxels: Vec::new(),
            })
            .collect();
        let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

        for lz in 0..4i32 {
            for ly in 0..4i32 {
                for lx in 0..4i32 {
                    if let Some(chunk) = self.octree.get_node_data_mut(
                        base_pos.x + lx,
                        base_pos.y + ly,
                        base_pos.z + lz,
                        0,
                    ) {
                        let mut modified = false;
                        let gx_base = (lx * 16) as usize;
                        let gy_base = (ly * 16) as usize;
                        let gz_base = (lz * 16) as usize;
                        let wx_base = ((base_pos.x + lx) * 64) as i32;
                        let wy_base = ((base_pos.y + ly) * 64) as i32;
                        let wz_base = ((base_pos.z + lz) * 64) as i32;

                        for z in 0..64usize {
                            let gz = gz_base + (z >> 2);
                            let gz_offset = gz * N2;
                            let wz = wz_base + z as i32;
                            for y in 0..64usize {
                                let col_idx = z * 64 + y;
                                let col = chunk.occupancy[col_idx];
                                if col == 0 {
                                    continue;
                                }
                                let gy = gy_base + (y >> 2);
                                let gy_offset = gz_offset + gy * N;
                                let wy = wy_base + y as i32;
                                let visual_row = z * 4096 + y * 64;
                                let mut col_new = col;
                                let mut col_remaining = col;
                                while col_remaining != 0 {
                                    let x = col_remaining.trailing_zeros() as usize;
                                    col_remaining &= col_remaining - 1;
                                    let gx = gx_base + (x >> 2);
                                    let label = coarse_grid[gy_offset + gx];
                                    if label >= 10 {
                                        let id = (label - 10) as usize;
                                        let wx = wx_base + x as i32;
                                        let mp = &mut min_points[id];
                                        mp.x = mp.x.min(wx);
                                        mp.y = mp.y.min(wy);
                                        mp.z = mp.z.min(wz);
                                        let visual_idx = visual_row + x;
                                        clump_results[id].voxels.push(FloatingVoxel {
                                            local_pos: Vector3i::new(wx, wy, wz),
                                            color_data: chunk.visuals[visual_idx],
                                        });
                                        col_new &= !(1u64 << x);
                                        chunk.visuals[visual_idx] = 0;
                                        modified = true;
                                    }
                                }
                                chunk.occupancy[col_idx] = col_new;
                            }
                        }
                        if modified {
                            chunk.is_dirty = true;
                            let key =
                                Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz);
                            affected_chunks.insert(key, key);
                        }
                    }
                }
            }
        }

        for i in 0..num_clumps {
            let m = min_points[i];
            clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
            for v in &mut clump_results[i].voxels {
                v.local_pos -= m;
            }
        }

        for cb in affected_chunks {
            let c = cb.0;
            self.texture_chunk_sparse(c.x, c.y, c.z, 0);
            self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
            self.mesh_chunk(c.x, c.y, c.z, false, 0, true, true);
        }

        clump_results
    }

    pub fn extract_floating_clumps_128(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
        let mut affected_chunks = HashMap::new();

        let mut coarse_grid = vec![0u8; 64 * 64 * 64];
        const N: usize = 64;
        const N2: usize = N * N;

        for lz in 0..2i32 {
            for ly in 0..2i32 {
                for lx in 0..2i32 {
                    let gx_base = (lx * 32) as usize; // 2 chunks * 32 coarse cells each = 64
                    let gy_base = (ly * 32) as usize;
                    let gz_base = (lz * 32) as usize;
                    if let Some(chunk) = self.octree.get_node_data_mut(
                        base_pos.x + lx,
                        base_pos.y + ly,
                        base_pos.z + lz,
                        0,
                    ) {
                        for z in 0..64usize {
                            let gz = gz_base + (z >> 1); // group 2 fine -> 1 coarse
                            let gz_offset = gz * N2;
                            let mut y = 0;
                            while y + 4 <= 64 {
                                let cols = u64x4::new([
                                    chunk.occupancy[z * 64 + y],
                                    chunk.occupancy[z * 64 + y + 1],
                                    chunk.occupancy[z * 64 + y + 2],
                                    chunk.occupancy[z * 64 + y + 3],
                                ]);
                                if cols == u64x4::ZERO {
                                    y += 4;
                                    continue;
                                }
                                for dy in 0..4 {
                                    let col = chunk.occupancy[z * 64 + y + dy];
                                    if col == 0 {
                                        continue;
                                    }
                                    let gy = gy_base + ((y + dy) >> 1); // group 2 fine -> 1 coarse
                                    let gy_offset = gz_offset + gy * N;
                                    // Collapse 2 fine bits -> 1 coarse bit
                                    // Mask picks bits at positions 0,2,4,... (every other)
                                    let mut bits = (col | (col >> 1)) & 0x5555_5555_5555_5555u64;
                                    while bits != 0 {
                                        let fx = bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        let gx = gx_base + (fx >> 1); // group 2 fine -> 1 coarse
                                        coarse_grid[gy_offset + gx] = 1;
                                    }
                                }
                                y += 4;
                            }
                        }
                    }
                }
            }
        }

        // Early exit: any interior solid? Interior = x/y/z in 1..63
        let has_interior_solid = {
            let mut found = false;
            'outer: for z in 1..63usize {
                let gz_off = z * N2;
                for y in 1..63usize {
                    let row_start = gz_off + y * N + 1; // x=1..62
                    let row = &coarse_grid[row_start..row_start + 62];
                    let mut i = 0;
                    while i + 8 <= row.len() {
                        let word = u64::from_ne_bytes(row[i..i + 8].try_into().unwrap());
                        if word != 0 {
                            found = true;
                            break 'outer;
                        }
                        i += 8;
                    }
                    while i < row.len() {
                        if row[i] != 0 {
                            found = true;
                            break 'outer;
                        }
                        i += 1;
                    }
                }
            }
            found
        };

        if !has_interior_solid {
            return Vec::new();
        }

        let mut stack: Vec<u32> = Vec::with_capacity(8192);
        {
            macro_rules! try_seed {
                ($idx:expr) => {
                    if coarse_grid[$idx] == 1 {
                        coarse_grid[$idx] = 2;
                        stack.push($idx as u32);
                    }
                };
            }
            for y in 0..N {
                for x in 0..N {
                    try_seed!(x + y * N);
                    try_seed!(x + y * N + 63 * N2);
                }
            }
            for z in 1..63usize {
                for x in 0..N {
                    try_seed!(x + z * N2);
                    try_seed!(x + 63 * N + z * N2);
                }
            }
            for z in 1..63usize {
                for y in 1..63usize {
                    try_seed!(y * N + z * N2);
                    try_seed!(63 + y * N + z * N2);
                }
            }
        }

        self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

        let mut next_id = 10u8;
        for i in 0..coarse_grid.len() {
            if coarse_grid[i] == 1 {
                coarse_grid[i] = next_id;
                stack.push(i as u32);
                self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
                next_id = next_id.saturating_add(1);
                if next_id == 255 {
                    break;
                }
            }
        }

        let num_clumps = (next_id - 10) as usize;
        if num_clumps == 0 {
            return Vec::new();
        }

        let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
            .map(|_| FloatingClump {
                world_origin: Vector3::ZERO,
                voxels: Vec::new(),
            })
            .collect();
        let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

        for lz in 0..2i32 {
            for ly in 0..2i32 {
                for lx in 0..2i32 {
                    if let Some(chunk) = self.octree.get_node_data_mut(
                        base_pos.x + lx,
                        base_pos.y + ly,
                        base_pos.z + lz,
                        0,
                    ) {
                        let mut modified = false;
                        let gx_base = (lx * 32) as usize;
                        let gy_base = (ly * 32) as usize;
                        let gz_base = (lz * 32) as usize;
                        let wx_base = ((base_pos.x + lx) * 64) as i32;
                        let wy_base = ((base_pos.y + ly) * 64) as i32;
                        let wz_base = ((base_pos.z + lz) * 64) as i32;

                        for z in 0..64usize {
                            let gz = gz_base + (z >> 1);
                            let gz_offset = gz * N2;
                            let wz = wz_base + z as i32;
                            for y in 0..64usize {
                                let col_idx = z * 64 + y;
                                let col = chunk.occupancy[col_idx];
                                if col == 0 {
                                    continue;
                                }
                                let gy = gy_base + (y >> 1);
                                let gy_offset = gz_offset + gy * N;
                                let wy = wy_base + y as i32;
                                let visual_row = z * 4096 + y * 64;
                                let mut col_new = col;
                                let mut col_remaining = col;
                                while col_remaining != 0 {
                                    let x = col_remaining.trailing_zeros() as usize;
                                    col_remaining &= col_remaining - 1;
                                    let gx = gx_base + (x >> 1);
                                    let label = coarse_grid[gy_offset + gx];
                                    if label >= 10 {
                                        let id = (label - 10) as usize;
                                        let wx = wx_base + x as i32;
                                        let mp = &mut min_points[id];
                                        mp.x = mp.x.min(wx);
                                        mp.y = mp.y.min(wy);
                                        mp.z = mp.z.min(wz);
                                        let visual_idx = visual_row + x;
                                        clump_results[id].voxels.push(FloatingVoxel {
                                            local_pos: Vector3i::new(wx, wy, wz),
                                            color_data: chunk.visuals[visual_idx],
                                        });
                                        col_new &= !(1u64 << x);
                                        chunk.visuals[visual_idx] = 0;
                                        modified = true;
                                    }
                                }
                                chunk.occupancy[col_idx] = col_new;
                            }
                        }
                        if modified {
                            chunk.is_dirty = true;
                            let key =
                                Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz);
                            affected_chunks.insert(key, key);
                        }
                    }
                }
            }
        }

        for i in 0..num_clumps {
            let m = min_points[i];
            clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
            for v in &mut clump_results[i].voxels {
                v.local_pos -= m;
            }
        }

        for cb in affected_chunks {
            let c = cb.0;
            self.texture_chunk_sparse(c.x, c.y, c.z, 0);
            self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
            self.mesh_chunk(c.x, c.y, c.z, false, 0, true, true);
        }

        clump_results
    }

    // pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
    //     let mut affected_chunks = HashMap::new();

    //     // --- PASS 1: COLLECT OCCUPANCY ---
    //     // Use u8 coarse grid. Also track whether ANY interior cell is solid —
    //     // if not, everything is grounded and we can return immediately.
    //     let mut coarse_grid = vec![0u8; 128 * 128 * 128];
    //     const N: usize = 128;
    //     const N2: usize = N * N;

    //     for lz in 0..4i32 {
    //         for ly in 0..4i32 {
    //             for lx in 0..4i32 {
    //                 let gx_base = (lx * 32) as usize;
    //                 let gy_base = (ly * 32) as usize;
    //                 let gz_base = (lz * 32) as usize;
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_offset = gz * N2;
    //                         let mut y = 0;
    //                         while y + 4 <= 64 {
    //                             let cols = u64x4::new([
    //                                 chunk.occupancy[z * 64 + y],
    //                                 chunk.occupancy[z * 64 + y + 1],
    //                                 chunk.occupancy[z * 64 + y + 2],
    //                                 chunk.occupancy[z * 64 + y + 3],
    //                             ]);
    //                             if cols == u64x4::ZERO {
    //                                 y += 4;
    //                                 continue;
    //                             }
    //                             for dy in 0..4 {
    //                                 let col = chunk.occupancy[z * 64 + y + dy];
    //                                 if col == 0 {
    //                                     continue;
    //                                 }
    //                                 let gy = gy_base + ((y + dy) >> 1);
    //                                 let gy_offset = gz_offset + gy * N;
    //                                 let mut bits = col;
    //                                 while bits != 0 {
    //                                     let x = bits.trailing_zeros() as usize;
    //                                     coarse_grid[gy_offset + gx_base + (x >> 1)] = 1;
    //                                     bits &= bits - 1;
    //                                     bits &= !(1u64 << (x ^ 1));
    //                                 }
    //                             }
    //                             y += 4;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- EARLY EXIT: scan interior for any solid cell ---
    //     // If every solid cell is on the border, nothing can be floating.
    //     // Scan interior slice (x/y/z in 1..127) using u64 words for speed.
    //     // We can skip the full grounded BFS entirely if this returns false.
    //     let has_interior_solid = {
    //         let mut found = false;
    //         'outer: for z in 1..127usize {
    //             let gz_off = z * N2;
    //             for y in 1..127usize {
    //                 let row_start = gz_off + y * N + 1; // x=1..126
    //                 // Check 2 u64 words covering x=1..126 (need ~2 words of 64 bits)
    //                 // row is 126 bytes; check word by word
    //                 let row = &coarse_grid[row_start..row_start + 126];
    //                 // Fast: check 8 bytes at a time as u64
    //                 let mut i = 0;
    //                 while i + 8 <= row.len() {
    //                     let word = u64::from_ne_bytes(row[i..i + 8].try_into().unwrap());
    //                     if word != 0 {
    //                         found = true;
    //                         break 'outer;
    //                     }
    //                     i += 8;
    //                 }
    //                 // Remaining bytes
    //                 while i < row.len() {
    //                     if row[i] != 0 {
    //                         found = true;
    //                         break 'outer;
    //                     }
    //                     i += 1;
    //                 }
    //             }
    //         }
    //         found
    //     };

    //     if !has_interior_solid {
    //         // Common case: fully grounded world. Zero BFS work needed.
    //         return Vec::new();
    //     }

    //     // --- PASS 2: SEED ANCHORS (only reached if interior solids exist) ---
    //     let mut stack: Vec<u32> = Vec::with_capacity(8192);
    //     {
    //         macro_rules! try_seed {
    //             ($idx:expr) => {
    //                 if coarse_grid[$idx] == 1 {
    //                     coarse_grid[$idx] = 2;
    //                     stack.push($idx as u32);
    //                 }
    //             };
    //         }
    //         for y in 0..N {
    //             for x in 0..N {
    //                 try_seed!(x + y * N);
    //                 try_seed!(x + y * N + 127 * N2);
    //             }
    //         }
    //         for z in 1..127usize {
    //             for x in 0..N {
    //                 try_seed!(x + z * N2);
    //                 try_seed!(x + 127 * N + z * N2);
    //             }
    //         }
    //         for z in 1..127usize {
    //             for y in 1..127usize {
    //                 try_seed!(y * N + z * N2);
    //                 try_seed!(127 + y * N + z * N2);
    //             }
    //         }
    //     }

    //     self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

    //     // --- PASS 3: LABEL INDIVIDUAL CLUMPS ---
    //     let mut next_id = 10u8;
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             coarse_grid[i] = next_id;
    //             stack.push(i as u32);
    //             self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
    //             next_id = next_id.saturating_add(1);
    //             if next_id == 255 {
    //                 break;
    //             }
    //         }
    //     }

    //     let num_clumps = (next_id - 10) as usize;
    //     if num_clumps == 0 {
    //         return Vec::new();
    //     }

    //     // --- PASS 4: EXTRACT ---
    //     let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
    //         .map(|_| FloatingClump {
    //             world_origin: Vector3::ZERO,
    //             voxels: Vec::new(),
    //         })
    //         .collect();
    //     let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

    //     for lz in 0..4i32 {
    //         for ly in 0..4i32 {
    //             for lx in 0..4i32 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let mut modified = false;
    //                     let gx_base = (lx * 32) as usize;
    //                     let gy_base = (ly * 32) as usize;
    //                     let gz_base = (lz * 32) as usize;
    //                     let wx_base = ((base_pos.x + lx) * 64) as i32;
    //                     let wy_base = ((base_pos.y + ly) * 64) as i32;
    //                     let wz_base = ((base_pos.z + lz) * 64) as i32;

    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_offset = gz * N2;
    //                         let wz = wz_base + z as i32;
    //                         for y in 0..64usize {
    //                             let col_idx = z * 64 + y;
    //                             let col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }
    //                             let gy = gy_base + (y >> 1);
    //                             let gy_offset = gz_offset + gy * N;
    //                             let wy = wy_base + y as i32;
    //                             let visual_row = z * 4096 + y * 64;
    //                             let mut col_new = col;
    //                             let mut col_remaining = col;
    //                             while col_remaining != 0 {
    //                                 let x = col_remaining.trailing_zeros() as usize;
    //                                 col_remaining &= col_remaining - 1;
    //                                 let gx = gx_base + (x >> 1);
    //                                 let label = coarse_grid[gy_offset + gx];
    //                                 if label >= 10 {
    //                                     let id = (label - 10) as usize;
    //                                     let wx = wx_base + x as i32;
    //                                     let mp = &mut min_points[id];
    //                                     mp.x = mp.x.min(wx);
    //                                     mp.y = mp.y.min(wy);
    //                                     mp.z = mp.z.min(wz);
    //                                     let visual_idx = visual_row + x;
    //                                     clump_results[id].voxels.push(FloatingVoxel {
    //                                         local_pos: Vector3i::new(wx, wy, wz),
    //                                         color_data: chunk.visuals[visual_idx],
    //                                     });
    //                                     col_new &= !(1u64 << x);
    //                                     chunk.visuals[visual_idx] = 0;
    //                                     modified = true;
    //                                 }
    //                             }
    //                             chunk.occupancy[col_idx] = col_new;
    //                         }
    //                     }
    //                     if modified {
    //                         chunk.is_dirty = true;
    //                         let key =
    //                             Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz);
    //                         affected_chunks.insert(key, key);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     for i in 0..num_clumps {
    //         let m = min_points[i];
    //         clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
    //         for v in &mut clump_results[i].voxels {
    //             v.local_pos -= m;
    //         }
    //     }

    //     for cb in affected_chunks {
    //         let c = cb.0;
    //         self.texture_chunk_sparse(c.x, c.y, c.z, 0);
    //         self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
    //         self.mesh_chunk(c.x, c.y, c.z, false, 0, true);
    //     }

    //     clump_results
    // }

    // pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
    //     // --- ALLOCATION OPTIMIZATIONS ---
    //     // Use u8 instead of u16 for coarse_grid - halves memory (2MB vs 4MB),
    //     // better cache utilization. We cap clump IDs at 245 (255-10).
    //     // Grounded=2, Air=0, Solid=1, Clump IDs=10..=255
    //     let mut coarse_grid = vec![0u8; 128 * 128 * 128];
    //     // Stack reused across both flood fills - avoids double allocation
    //     let mut stack: Vec<u32> = Vec::with_capacity(8192);

    //     let mut affected_chunks = HashMap::new();

    //     // --- PASS 1: COLLECT OCCUPANCY ---
    //     // Precompute chunk-local offsets once, not per-voxel
    //     for lz in 0..4i32 {
    //         for ly in 0..4i32 {
    //             for lx in 0..4i32 {
    //                 let gx_base = (lx * 32) as usize;
    //                 let gy_base = (ly * 32) as usize;
    //                 let gz_base = (lz * 32) as usize;

    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_offset = gz * 128 * 128;

    //                         // Process 4 y-columns simultaneously
    //                         let mut y = 0;
    //                         while y + 4 <= 64 {
    //                             let cols = u64x4::new([
    //                                 chunk.occupancy[z * 64 + y],
    //                                 chunk.occupancy[z * 64 + y + 1],
    //                                 chunk.occupancy[z * 64 + y + 2],
    //                                 chunk.occupancy[z * 64 + y + 3],
    //                             ]);

    //                             // Early exit if all 4 columns are empty
    //                             if cols == u64x4::ZERO {
    //                                 y += 4;
    //                                 continue;
    //                             }

    //                             // Fall back to scalar for non-empty (columns are usually sparse)
    //                             for dy in 0..4 {
    //                                 let col = chunk.occupancy[z * 64 + y + dy];
    //                                 if col == 0 {
    //                                     continue;
    //                                 }
    //                                 let gy = gy_base + ((y + dy) >> 1);
    //                                 let gy_offset = gz_offset + gy * 128;
    //                                 let mut bits = col;
    //                                 while bits != 0 {
    //                                     let x = bits.trailing_zeros() as usize;
    //                                     coarse_grid[gy_offset + gx_base + (x >> 1)] = 1;
    //                                     bits &= bits - 1;
    //                                     bits &= !(1u64 << (x ^ 1)); // clear sibling bit too
    //                                 }
    //                             }
    //                             y += 4;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- PASS 2: SEED ANCHORS ---
    //     // Instead of iterating all 128^3 cells to find borders,
    //     // only iterate the 6 faces directly.
    //     {
    //         // Helper closure to seed a cell
    //         macro_rules! try_seed {
    //             ($idx:expr) => {
    //                 if coarse_grid[$idx] == 1 {
    //                     coarse_grid[$idx] = 2;
    //                     stack.push($idx as u32);
    //                 }
    //             };
    //         }

    //         const N: usize = 128;
    //         const N2: usize = N * N;

    //         // Face Z=0 and Z=127
    //         for y in 0..N {
    //             for x in 0..N {
    //                 try_seed!(x + y * N); // z=0
    //                 try_seed!(x + y * N + 127 * N2); // z=127
    //             }
    //         }
    //         // Face Y=0 and Y=127 (excluding corners already done)
    //         for z in 1..127 {
    //             for x in 0..N {
    //                 try_seed!(x + 0 * N + z * N2); // y=0
    //                 try_seed!(x + 127 * N + z * N2); // y=127
    //             }
    //         }
    //         // Face X=0 and X=127 (excluding edges already done)
    //         for z in 1..127 {
    //             for y in 1..127 {
    //                 try_seed!(0 + y * N + z * N2); // x=0
    //                 try_seed!(127 + y * N + z * N2); // x=127
    //             }
    //         }
    //     }

    //     // Propagate grounded signal
    //     self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

    //     // --- PASS 3: LABEL INDIVIDUAL CLUMPS ---
    //     let mut next_id = 10u8;
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             coarse_grid[i] = next_id;
    //             stack.push(i as u32);
    //             self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
    //             next_id = next_id.saturating_add(1);
    //             // Guard against overflow (max 245 clumps with u8)
    //             if next_id == 255 {
    //                 break;
    //             }
    //         }
    //     }

    //     // --- PASS 4: PURGE AND EXTRACT ---
    //     let num_clumps = (next_id - 10) as usize;
    //     let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
    //         .map(|_| FloatingClump {
    //             world_origin: Vector3::ZERO,
    //             voxels: Vec::new(),
    //         })
    //         .collect();
    //     let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

    //     for lz in 0..4i32 {
    //         for ly in 0..4i32 {
    //             for lx in 0..4i32 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let mut modified = false;
    //                     let gx_base = (lx * 32) as usize;
    //                     let gy_base = (ly * 32) as usize;
    //                     let gz_base = (lz * 32) as usize;
    //                     let wx_base = ((base_pos.x + lx) * 64) as i32;
    //                     let wy_base = ((base_pos.y + ly) * 64) as i32;
    //                     let wz_base = ((base_pos.z + lz) * 64) as i32;

    //                     for z in 0..64usize {
    //                         let gz = gz_base + (z >> 1);
    //                         let gz_offset = gz * 128 * 128;
    //                         let wz = wz_base + z as i32;

    //                         for y in 0..64usize {
    //                             let col_idx = z * 64 + y;
    //                             let col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }

    //                             let gy = gy_base + (y >> 1);
    //                             let gy_offset = gz_offset + gy * 128;
    //                             let wy = wy_base + y as i32;
    //                             let visual_row = z * 4096 + y * 64; // base of visual row

    //                             let mut col_mut = col;
    //                             let mut col_new = col;

    //                             let mut col_remaining = col;
    //                             while col_remaining != 0 {
    //                                 let x = col_remaining.trailing_zeros() as usize;
    //                                 col_remaining &= col_remaining - 1; // clear lowest set bit

    //                                 let gx = gx_base + (x >> 1);
    //                                 let label = coarse_grid[gy_offset + gx];

    //                                 if label >= 10 {
    //                                     let id = (label - 10) as usize;
    //                                     let wx = wx_base + x as i32;

    //                                     // Branchless min update
    //                                     let mp = &mut min_points[id];
    //                                     mp.x = mp.x.min(wx);
    //                                     mp.y = mp.y.min(wy);
    //                                     mp.z = mp.z.min(wz);

    //                                     let visual_idx = visual_row + x;
    //                                     clump_results[id].voxels.push(FloatingVoxel {
    //                                         local_pos: Vector3i::new(wx, wy, wz),
    //                                         color_data: chunk.visuals[visual_idx],
    //                                     });

    //                                     col_new &= !(1u64 << x);
    //                                     chunk.visuals[visual_idx] = 0;
    //                                     modified = true;
    //                                 }
    //                             }

    //                             chunk.occupancy[col_idx] = col_new;
    //                         }
    //                     }

    //                     if modified {
    //                         chunk.is_dirty = true;
    //                         let key =
    //                             Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz);
    //                         affected_chunks.insert(key, key);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Finalize clump offsets
    //     for i in 0..num_clumps {
    //         let m = min_points[i];
    //         clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
    //         for v in &mut clump_results[i].voxels {
    //             v.local_pos -= m;
    //         }
    //     }

    //     for cb in affected_chunks {
    //         let c = cb.0;
    //         self.texture_chunk_sparse(c.x, c.y, c.z, 0);
    //         self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
    //         self.mesh_chunk(c.x, c.y, c.z, false, 0, true);
    //     }

    //     clump_results
    // }

    fn flood_fill_coarse_256(
        &self,
        grid: &mut [u8],      // Changed from u16 to u8
        stack: &mut Vec<u32>, // Pack (x,y,z) into a single u32 index
        fill_id: u8,
    ) {
        const N: i32 = 64;
        const N2: i32 = N * N;

        while let Some(idx) = stack.pop() {
            let idx = idx as i32;
            let x = idx % N;
            let y = (idx / N) % N;
            let z = idx / N2;

            // Unrolled neighbors with bounds check fused into index arithmetic
            // Each neighbor: check bounds, then check grid value
            macro_rules! check_neighbor {
                ($nx:expr, $ny:expr, $nz:expr) => {
                    let nx = $nx;
                    let ny = $ny;
                    let nz = $nz;
                    if nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N {
                        let n_idx = (nz * N2 + ny * N + nx) as usize;
                        if grid[n_idx] == 1 {
                            grid[n_idx] = fill_id;
                            stack.push(n_idx as u32);
                        }
                    }
                };
            }

            check_neighbor!(x + 1, y, z);
            check_neighbor!(x - 1, y, z);
            check_neighbor!(x, y + 1, z);
            check_neighbor!(x, y - 1, z);
            check_neighbor!(x, y, z + 1);
            check_neighbor!(x, y, z - 1);
        }
    }

    // pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
    //     // 128^3 coarse grid (4MB). 0=Air, 1=Solid, 2=Grounded, 10+=Clump ID
    //     let mut coarse_grid = vec![0u16; 128 * 128 * 128];
    //     let mut stack = Vec::with_capacity(4096);

    //     let mut affected_chunks = HashMap::new();

    //     // --- PASS 1: COLLECT OCCUPANCY (READ ONLY) ---
    //     for lz in 0..4 {
    //         for ly in 0..4 {
    //             for lx in 0..4 {
    //                 let cx = base_pos.x + lx;
    //                 let cy = base_pos.y + ly;
    //                 let cz = base_pos.z + lz;

    //                 if let Some(chunk) = self.octree.get_node_data_mut(cx, cy, cz, 0) {
    //                     // FIXED LAYOUT: Z and Y are the array indices, X is the bit
    //                     for z in 0..64 {
    //                         for y in 0..64 {
    //                             let col_idx = (z * 64) + y;
    //                             let col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }
    //                             for x in 0..64 {
    //                                 if (col >> x) & 1 == 1 {
    //                                     // Map to the global coarse grid (0-127)
    //                                     let gx = (lx * 32) + (x as i32 / 2);
    //                                     let gy = (ly * 32) + (y as i32 / 2);
    //                                     let gz = (lz * 32) + (z as i32 / 2);
    //                                     let idx = (gz * 16384) + (gy * 128) + gx;
    //                                     coarse_grid[idx as usize] = 1;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // --- PASS 2: SEED ANCHORS (OUTER SHELL ONLY) ---
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             let gx = i % 128;
    //             let gy = (i / 128) % 128;
    //             let gz = i / 16384;

    //             // ONLY anchor to the boundaries of the 256x256x256 volume.
    //             if gx == 0 || gx == 127 || gy == 0 || gy == 127 || gz == 0 || gz == 127 {
    //                 coarse_grid[i] = 2;
    //                 stack.push((gx as i32, gy as i32, gz as i32));
    //             }
    //         }
    //     }

    //     // Propagate "Grounded" signal
    //     self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

    //     // --- PASS 3: LABEL INDIVIDUAL CLUMPS ---
    //     let mut next_id = 10u16;
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             // This voxel is solid but NOT grounded
    //             let gx = (i % 128) as i32;
    //             let gy = ((i / 128) % 128) as i32;
    //             let gz = (i / 16384) as i32;

    //             stack.push((gx, gy, gz));
    //             coarse_grid[i] = next_id;
    //             self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
    //             next_id += 1;
    //         }
    //     }

    //     // --- PASS 4: PURGE AND EXTRACT (WRITE) ---
    //     let num_clumps = (next_id - 10) as usize;
    //     let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
    //         .map(|_| FloatingClump {
    //             world_origin: Vector3::ZERO,
    //             voxels: Vec::new(),
    //         })
    //         .collect();
    //     let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

    //     for lz in 0..4 {
    //         for ly in 0..4 {
    //             for lx in 0..4 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let mut modified = false;

    //                     // FIXED LAYOUT: Z and Y are the array indices, X is the bit
    //                     for z in 0..64 {
    //                         for y in 0..64 {
    //                             let col_idx = (z * 64) + y;
    //                             let mut col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }

    //                             for x in 0..64 {
    //                                 if (col >> x) & 1 == 1 {
    //                                     let gx = (lx * 32) + (x as i32 / 2);
    //                                     let gy = (ly * 32) + (y as i32 / 2);
    //                                     let gz = (lz * 32) + (z as i32 / 2);
    //                                     let label =
    //                                         coarse_grid[(gz * 16384 + gy * 128 + gx) as usize];

    //                                     if label >= 10 {
    //                                         let id = (label - 10) as usize;
    //                                         let wx = ((base_pos.x + lx) * 64) + x as i32;
    //                                         let wy = ((base_pos.y + ly) * 64) + y as i32;
    //                                         let wz = ((base_pos.z + lz) * 64) + z as i32;

    //                                         min_points[id].x = min_points[id].x.min(wx);
    //                                         min_points[id].y = min_points[id].y.min(wy);
    //                                         min_points[id].z = min_points[id].z.min(wz);

    //                                         // Visuals index calculation matching set_voxel_full
    //                                         let visual_idx = (z as usize * 4096)
    //                                             + (y as usize * 64)
    //                                             + x as usize;

    //                                         clump_results[id].voxels.push(FloatingVoxel {
    //                                             local_pos: Vector3i::new(wx, wy, wz),
    //                                             color_data: chunk.visuals[visual_idx],
    //                                         });

    //                                         col &= !(1u64 << x); // Using 1u64 to prevent shifting overflow bugs
    //                                         chunk.visuals[visual_idx] = 0;
    //                                         modified = true;
    //                                     }
    //                                 }
    //                             }
    //                             chunk.occupancy[col_idx] = col;
    //                         }
    //                     }
    //                     if modified {
    //                         chunk.is_dirty = true;
    //                         affected_chunks.insert(
    //                             Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz),
    //                             Vector3i::new(base_pos.x + lx, base_pos.y + ly, base_pos.z + lz),
    //                         );
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Finalize clump offsets
    //     for i in 0..num_clumps {
    //         let m = min_points[i];
    //         clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
    //         for v in &mut clump_results[i].voxels {
    //             v.local_pos -= m;
    //         }
    //     }

    //     for cb in affected_chunks {
    //         let c = cb.0;
    //         self.texture_chunk_sparse(c.x, c.y, c.z, 0);
    //         self.update_chunk_texture_sparse(c.x, c.y, c.z, 0);
    //         self.mesh_chunk(c.x, c.y, c.z, false, 0, true);
    //     }

    //     clump_results
    // }

    // fn flood_fill_coarse_256(
    //     &self,
    //     grid: &mut [u16],
    //     stack: &mut Vec<(i32, i32, i32)>,
    //     fill_id: u16,
    // ) {
    //     while let Some((x, y, z)) = stack.pop() {
    //         for (dx, dy, dz) in [
    //             (1, 0, 0),
    //             (-1, 0, 0),
    //             (0, 1, 0),
    //             (0, -1, 0),
    //             (0, 0, 1),
    //             (0, 0, -1),
    //         ] {
    //             let nx = x + dx;
    //             let ny = y + dy;
    //             let nz = z + dz;
    //             if nx >= 0 && nx < 128 && ny >= 0 && ny < 128 && nz >= 0 && nz < 128 {
    //                 let n_idx = (nz as usize * 128 * 128) + (ny as usize * 128) + nx as usize;
    //                 if grid[n_idx] == 1 {
    //                     grid[n_idx] = fill_id;
    //                     stack.push((nx, ny, nz));
    //                 }
    //             }
    //         }
    //     }
    // }

    // pub fn extract_floating_clumps_256(&mut self, base_pos: Vector3i) -> Vec<FloatingClump> {
    //     // 128^3 coarse grid (4MB). 0=Air, 1=Solid, 2=Grounded, 10+=Clump ID
    //     let mut coarse_grid = vec![0u16; 128 * 128 * 128];
    //     let mut stack = Vec::with_capacity(4096);

    //     // --- PASS 1: COLLECT OCCUPANCY (READ ONLY) ---
    //     for lz in 0..4 {
    //         for ly in 0..4 {
    //             for lx in 0..4 {
    //                 let cx = base_pos.x + lx;
    //                 let cy = base_pos.y + ly;
    //                 let cz = base_pos.z + lz;

    //                 // Use your only available accessor
    //                 if let Some(chunk) = self.octree.get_node_data_mut(cx, cy, cz, 0) {
    //                     for y in 0..64 {
    //                         for x in 0..64 {
    //                             let col = chunk.occupancy[(y * 64) + x];
    //                             if col == 0 {
    //                                 continue;
    //                             }
    //                             for z in 0..64 {
    //                                 if (col >> z) & 1 == 1 {
    //                                     // Map to the global coarse grid (0-127)
    //                                     let gx = (lx * 32) + (x as i32 / 2);
    //                                     let gy = (ly * 32) + (y as i32 / 2);
    //                                     let gz = (lz * 32) + (z / 2);
    //                                     let idx = (gz * 16384) + (gy * 128) + gx;
    //                                     coarse_grid[idx as usize] = 1;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //                 // Note: If chunk is None, it remains 0 (Air) in coarse_grid.
    //             }
    //         }
    //     }

    //     // --- PASS 2: SEED ANCHORS (OUTER SHELL ONLY) ---
    //     // We iterate the coarse grid. If a voxel is at the VERY edge of the 4x4x4 area, it's Grounded.
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             let gx = i % 128;
    //             let gy = (i / 128) % 128;
    //             let gz = i / 16384;

    //             // ONLY anchor to the boundaries of the 256x256x256 volume.
    //             // Internal chunk borders (e.g., gx=31/32) are ignored here.
    //             if gx == 0 || gx == 127 || gy == 0 || gy == 127 || gz == 0 || gz == 127 {
    //                 coarse_grid[i] = 2;
    //                 stack.push((gx as i32, gy as i32, gz as i32));
    //             }
    //         }
    //     }

    //     // Propagate "Grounded" signal
    //     self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, 2);

    //     // --- PASS 3: LABEL INDIVIDUAL CLUMPS ---
    //     let mut next_id = 10u16;
    //     for i in 0..coarse_grid.len() {
    //         if coarse_grid[i] == 1 {
    //             // This voxel is solid but NOT grounded
    //             let gx = (i % 128) as i32;
    //             let gy = ((i / 128) % 128) as i32;
    //             let gz = (i / 16384) as i32;

    //             stack.push((gx, gy, gz));
    //             coarse_grid[i] = next_id;
    //             self.flood_fill_coarse_256(&mut coarse_grid, &mut stack, next_id);
    //             next_id += 1;
    //         }
    //     }

    //     // --- PASS 4: PURGE AND EXTRACT (WRITE) ---
    //     let num_clumps = (next_id - 10) as usize;
    //     let mut clump_results: Vec<FloatingClump> = (0..num_clumps)
    //         .map(|_| FloatingClump {
    //             world_origin: Vector3::ZERO,
    //             voxels: Vec::new(),
    //         })
    //         .collect();
    //     let mut min_points = vec![Vector3i::new(i32::MAX, i32::MAX, i32::MAX); num_clumps];

    //     for lz in 0..4 {
    //         for ly in 0..4 {
    //             for lx in 0..4 {
    //                 if let Some(chunk) = self.octree.get_node_data_mut(
    //                     base_pos.x + lx,
    //                     base_pos.y + ly,
    //                     base_pos.z + lz,
    //                     0,
    //                 ) {
    //                     let mut modified = false;
    //                     for y in 0..64 {
    //                         for x in 0..64 {
    //                             let col_idx = (y * 64) + x;
    //                             let mut col = chunk.occupancy[col_idx];
    //                             if col == 0 {
    //                                 continue;
    //                             }

    //                             for z in 0..64 {
    //                                 if (col >> z) & 1 == 1 {
    //                                     let gx = (lx * 32) + (x as i32 / 2);
    //                                     let gy = (ly * 32) + (y as i32 / 2);
    //                                     let gz = (lz * 32) + (z / 2);
    //                                     let label =
    //                                         coarse_grid[(gz * 16384 + gy * 128 + gx) as usize];

    //                                     if label >= 10 {
    //                                         let id = (label - 10) as usize;
    //                                         let wx = ((base_pos.x + lx) * 64) + x as i32;
    //                                         let wy = ((base_pos.y + ly) * 64) + y as i32;
    //                                         let wz = ((base_pos.z + lz) * 64) + z as i32;

    //                                         min_points[id].x = min_points[id].x.min(wx);
    //                                         min_points[id].y = min_points[id].y.min(wy);
    //                                         min_points[id].z = min_points[id].z.min(wz);

    //                                         let col_idx_u = col_idx as usize;
    //                                         let z_u = z as usize;

    //                                         // 2. Calculate the flat index
    //                                         let visual_idx = (z_u * 4096) + col_idx_u;

    //                                         clump_results[id].voxels.push(FloatingVoxel {
    //                                             local_pos: Vector3i::new(wx, wy, wz),
    //                                             // 3. Use visual_idx to index into the array
    //                                             color_data: chunk.visuals[visual_idx],
    //                                         });

    //                                         col &= !(1 << z);
    //                                         chunk.visuals[visual_idx] = 0;
    //                                         modified = true;
    //                                     }
    //                                 }
    //                             }
    //                             chunk.occupancy[col_idx] = col;
    //                         }
    //                     }
    //                     if modified {
    //                         chunk.is_dirty = true;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Finalize clump offsets
    //     for i in 0..num_clumps {
    //         let m = min_points[i];
    //         clump_results[i].world_origin = Vector3::new(m.x as f32, m.y as f32, m.z as f32);
    //         for v in &mut clump_results[i].voxels {
    //             v.local_pos -= m;
    //         }
    //     }

    //     clump_results
    // }

    // fn flood_fill_coarse_256(
    //     &self,
    //     grid: &mut [u16],
    //     stack: &mut Vec<(i32, i32, i32)>,
    //     fill_id: u16,
    // ) {
    //     while let Some((x, y, z)) = stack.pop() {
    //         for (dx, dy, dz) in [
    //             (1, 0, 0),
    //             (-1, 0, 0),
    //             (0, 1, 0),
    //             (0, -1, 0),
    //             (0, 0, 1),
    //             (0, 0, -1),
    //         ] {
    //             let nx = x + dx;
    //             let ny = y + dy;
    //             let nz = z + dz;
    //             if nx >= 0 && nx < 128 && ny >= 0 && ny < 128 && nz >= 0 && nz < 128 {
    //                 let n_idx = (nz as usize * 128 * 128) + (ny as usize * 128) + nx as usize;
    //                 if grid[n_idx] == 1 {
    //                     grid[n_idx] = fill_id;
    //                     stack.push((nx, ny, nz));
    //                 }
    //             }
    //         }
    //     }
    // }

    // ─────────────────────────────────────────────────────────────────────────────
    //  update_lod_streaming
    //  Called once per frame.  Rebuilds the work queue from the octree top-down.
    // ─────────────────────────────────────────────────────────────────────────────
    #[func]
    pub fn update_lod_streaming(&mut self, camera_pos: Vector3) {
        // The physical size of one LOD-0 chunk side in world units
        let chunk_world = 64.0 * self.voxel_size;

        // Camera position in LOD-0 chunk coordinates
        let cam_cx = (camera_pos.x / chunk_world).floor() as i32;
        let cam_cy = (camera_pos.y / chunk_world).floor() as i32;
        let cam_cz = (camera_pos.z / chunk_world).floor() as i32;

        let cam_chunk = Vector3i::new(cam_cx, cam_cy, cam_cz);

        // Only re-scan if the camera has moved to a new chunk
        if cam_chunk == self.world_state.last_camera_chunk {
            return;
        }
        self.world_state.last_camera_chunk = cam_chunk;

        // Collect the new desired set of (coord, depth) pairs via octree subdivision.
        // We store them in a temporary Vec so we can diff against loaded_chunks.
        let mut desired: Vec<WorkItem> = Vec::with_capacity(4096);

        // The side length (in LOD-0 chunks) of one LOD-4 super-node
        let lod4_stride = 1_i32 << 4; // = 16

        // Scan a cube of LOD-4 nodes around the camera
        let half = self.lod_scan_half_extent;
        let root_cx = cam_cx >> 4; // which LOD-4 column the camera is in
        let root_cy = cam_cy >> 4;
        let root_cz = cam_cz >> 4;

        for rx in (root_cx - half)..=(root_cx + half) {
            for ry in (root_cy - half)..=(root_cy + half) {
                for rz in (root_cz - half)..=(root_cz + half) {
                    // The LOD-0 origin of this LOD-4 node
                    let origin_x = rx * lod4_stride;
                    let origin_y = ry * lod4_stride;
                    let origin_z = rz * lod4_stride;

                    // Recursively subdivide, starting at depth 4
                    self.subdivide_node(
                        camera_pos,
                        chunk_world,
                        origin_x,
                        origin_y,
                        origin_z,
                        4, // starting depth
                        &mut desired,
                    );
                }
            }
        }

        // Diff desired against currently loaded chunks.
        // Anything new → push to update_queue.
        // (Unloading out-of-range chunks is a separate concern handled elsewhere.)
        self.world_state.update_queue.clear();

        for item in &desired {
            let key = (item.coord, item.depth);

            let already_loaded = self.world_state.loaded_chunks.contains_key(&key);

            if !already_loaded {
                self.world_state.update_queue.push_back(WorkItem {
                    coord: item.coord,
                    depth: item.depth,
                });
                self.world_state.loaded_chunks.insert(key, true);
            }
        }

        // Priority sort: depth-0 (HD) first, then by ascending distance.
        // LOD chunks get a large distance penalty so they always trail HD work.
        let center = cam_chunk;
        let mut temp: Vec<WorkItem> = self.world_state.update_queue.drain(..).collect();
        temp.sort_by_cached_key(|item| {
            let dx = item.coord.x - center.x;
            let dy = item.coord.y - center.y;
            let dz = item.coord.z - center.z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if item.depth == 0 {
                dist_sq
            } else {
                // Push LOD chunks after all HD work; coarser = further back
                dist_sq + 2_000_000 * item.depth as i32
            }
        });
        self.world_state.update_queue.extend(temp);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  subdivide_node  (private recursive helper)
    //
    //  Determines whether a node at (ox, oy, oz) at `depth` should be rendered
    //  at this LOD level or subdivided further.
    //
    //  `ox/oy/oz` are the LOD-0-space coordinates of the node's corner.
    //  The node covers a cube of side `1 << depth` in LOD-0 chunk units.
    // ─────────────────────────────────────────────────────────────────────────────
    fn subdivide_node(
        &self,
        camera_pos: Vector3,
        chunk_world: f32,
        ox: i32,
        oy: i32,
        oz: i32,
        depth: u8,
        out: &mut Vec<WorkItem>,
    ) {
        let stride = 1_i32 << depth; // side length in LOD-0 chunks

        // World-space centre of this node
        let half = stride as f32 * 0.5;
        let centre_world = Vector3::new(
            (ox as f32 + half) * chunk_world,
            (oy as f32 + half) * chunk_world,
            (oz as f32 + half) * chunk_world,
        );

        let dist = (centre_world - camera_pos).length();

        // At depth 0 we are at a leaf — always enqueue for generation.
        if depth == 0 {
            out.push(WorkItem {
                coord: Vector3i::new(ox, oy, oz),
                depth: 0,
            });
            return;
        }

        // Read the configured split radius for this depth level.
        // lod_radii[depth] is the world-unit distance inside which we subdivide.
        let split_radius = self.lod_radii.get(depth as usize).unwrap_or(f32::MAX);

        if dist <= split_radius {
            // ── SUBDIVIDE: recurse into 8 children at (depth - 1) ──────────────
            let child_stride = stride >> 1;
            for i in 0..8_usize {
                let cx = ox + ((i & 1) as i32) * child_stride;
                let cy = oy + (((i >> 1) & 1) as i32) * child_stride;
                let cz = oz + (((i >> 2) & 1) as i32) * child_stride;
                self.subdivide_node(camera_pos, chunk_world, cx, cy, cz, depth - 1, out);
            }
        } else {
            // ── LEAF AT THIS LOD: render the whole node as one coarse chunk ─────
            // The chunk coordinate for a depth-N node is the LOD-0 coordinate of
            // its corner, right-shifted by N  (matching get_or_create_node usage).
            let chunk_coord = Vector3i::new(ox >> depth, oy >> depth, oz >> depth);
            out.push(WorkItem {
                coord: chunk_coord,
                depth,
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  process_lod_queue
    //  Called once per frame after update_lod_streaming.
    //  Drains up to `max_hd_jobs` depth-0 jobs and `max_lod_jobs` LOD jobs,
    //  subject to a `frame_limit_ms` wall-clock budget.
    // ─────────────────────────────────────────────────────────────────────────────
    #[func]
    pub fn process_lod_queue(&mut self, max_hd_jobs: i32, max_lod_jobs: i32, frame_limit_ms: f32) {
        let time_singleton = godot::classes::Time::singleton();
        let frame_start = time_singleton.get_ticks_usec();
        let budget_usec = (frame_limit_ms * 1000.0) as u64;

        let mut hd_done = 0_i32;
        let mut lod_done = 0_i32;

        loop {
            // ── Time guard ────────────────────────────────────────────────────
            let elapsed = time_singleton.get_ticks_usec().saturating_sub(frame_start);
            if elapsed > budget_usec {
                break;
            }

            // ── Quota guard ───────────────────────────────────────────────────
            if hd_done >= max_hd_jobs && lod_done >= max_lod_jobs {
                break;
            }

            let job = match self.world_state.update_queue.pop_front() {
                Some(j) => j,
                None => break,
            };

            let is_hd = job.depth == 0;

            if is_hd && hd_done >= max_hd_jobs {
                // Re-queue and look for a LOD job instead
                self.world_state.update_queue.push_front(job);
                // Skip to next iteration hoping to find a LOD job
                // (avoids starving LOD work when HD quota is full)
                if lod_done >= max_lod_jobs {
                    break;
                }
                continue;
            }
            if !is_hd && lod_done >= max_lod_jobs {
                continue; // skip LOD jobs when LOD quota full, keep draining HDs
            }

            let (cx, cy, cz) = (job.coord.x, job.coord.y, job.coord.z);
            let cache_key = (job.coord, job.depth);

            let already_exists = self
                .octree
                .get_or_create_node(cx, cy, cz, job.depth, false)
                .and_then(|idx| self.octree.node_pool[idx as usize].chunk.as_ref())
                .is_some();

            if already_exists {
                continue;
            }

            // ── Coarse cache check ────────────────────────────────────────────
            let cached_state = self
                .terrain_manager
                .as_ref()
                .map_or(ChunkState::Unknown, |tm| {
                    tm.bind()
                        .coarse_cache
                        .get(&cache_key)
                        .cloned()
                        .unwrap_or(ChunkState::Unknown)
                });

            if cached_state == ChunkState::Air || cached_state == ChunkState::Solid {
                // Known-boring chunk — skip instantly, doesn't count as a job
                continue;
            }

            // ── Coarse data preparation ───────────────────────────────────────
            self.prepare_chunk_data(cx, cy, cz, job.depth as i32);
            let mixed = self.is_chunk_mixed(cx, cy, cz, job.depth as i32);

            // Update the coarse cache with what we just learned
            if let Some(mut tm_bind) = self.terrain_manager.as_mut().map(|tm| tm.bind_mut()) {
                tm_bind.coarse_cache.insert(
                    cache_key,
                    if mixed {
                        ChunkState::Mixed
                    } else {
                        ChunkState::Air
                    },
                );
            }

            if !mixed {
                // Turned out to be empty/boring after sampling — skip, no job slot used
                continue;
            }

            // ── Full generation pipeline ──────────────────────────────────────
            self.create_chunk(cx, cy, cz, job.depth, true);
            self.generate_chunk_3d(cx, cy, cz, job.depth as i32);
            self.texture_chunk_sparse(cx, cy, cz, job.depth);
            self.update_chunk_texture_sparse(cx, cy, cz, job.depth);
            self.mesh_chunk(cx, cy, cz, false, job.depth, true, false);

            if is_hd {
                hd_done += 1;
            } else {
                lod_done += 1;
            }
        }
    }
}

//
// helpers
//
const N: usize = 64;
const N2: usize = N * N;
const N3: usize = N * N * N;
const GROUND: u32 = N3 as u32; // sentinel root for "grounded" nodes
const MAX_LABELS: usize = 4096;

#[inline(always)]
fn label_find(table: &mut [u16; MAX_LABELS], mut i: u16) -> u16 {
    // Path-halving on a 4096-entry array — entirely L1 resident
    while table[i as usize] != i {
        let g = table[table[i as usize] as usize];
        table[i as usize] = g;
        i = g;
    }
    i
}

#[inline(always)]
fn label_union(table: &mut [u16; MAX_LABELS], a: u16, b: u16) {
    let ra = label_find(table, a);
    let rb = label_find(table, b);
    if ra != rb {
        // Always let lower ID win — keeps GROUND=0 as the sticky root
        if ra < rb {
            table[rb as usize] = ra;
        } else {
            table[ra as usize] = rb;
        }
    }
}

const CHUNKS: usize = 64; // 4*4*4
const COL_COUNT: usize = CHUNKS * 64 * 64; // 262144 column nodes
const GROUND_NODE: u32 = COL_COUNT as u32; // sentinel

struct ClumpFinder {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl ClumpFinder {
    fn new() -> Self {
        let n = COL_COUNT + 1;
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0u8; n],
        }
    }

    #[inline(always)]
    fn find(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            let g = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = g;
            i = g;
        }
        i
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        // GROUND always wins
        if rb == GROUND_NODE {
            std::mem::swap(&mut ra, &mut rb);
        }
        if ra == GROUND_NODE {
            self.parent[rb as usize] = GROUND_NODE;
            return;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
    }
}

// chunk index from (lx,ly,lz) in 0..4
#[inline(always)]
fn chunk_idx(lx: usize, ly: usize, lz: usize) -> usize {
    lz * 16 + ly * 4 + lx
}

// column node ID from chunk_idx, z, y
#[inline(always)]
fn col_node(cidx: usize, z: usize, y: usize) -> u32 {
    (cidx * 64 * 64 + z * 64 + y) as u32
}
