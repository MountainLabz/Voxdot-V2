use godot::classes::Image;
use godot::classes::Texture2D;
use godot::classes::image::Format;
use godot::classes::{Curve, Material, Resource};
use godot::prelude::*;
use std::collections::HashMap;

use fastnoise2::SafeNode;
//use hashbrown::HashMap;

use hashbrown::HashSet;
use std::collections::VecDeque;

pub struct WorkItem {
    pub coord: Vector3i,
    pub depth: u8,
}

pub struct LoadedWorld {
    pub loaded_chunks: HashMap<(Vector3i, u8), bool>,
    pub update_queue: VecDeque<WorkItem>,
    pub last_camera_chunk: Vector3i,
    pub current_scan_x: i32,
}

impl LoadedWorld {
    pub fn new() -> Self {
        Self {
            loaded_chunks: HashMap::new(),
            update_queue: VecDeque::new(),
            last_camera_chunk: Vector3i::new(i32::MAX, i32::MAX, i32::MAX),
            current_scan_x: -100,
        }
    }
}

pub struct GpuUploadData {
    pub bitmask: [u64; 4096],     // 64x64 columns
    pub index_map: [u32; 4096],   // 64x64 prefix sums
    pub sparse_visuals: Vec<u32>, // The actual 32-bit voxels
    pub voxel_count: u32,
}

impl GpuUploadData {
    pub fn new() -> Self {
        Self {
            bitmask: [0u64; 4096],
            index_map: [0u32; 4096],
            sparse_visuals: Vec::with_capacity(4096), // Pre-allocate to avoid early reallocs
            voxel_count: 0,
        }
    }
}

// --- Updated Noise Cache using SafeNode ---
pub struct NoiseCache {
    pub compiled_nodes: HashMap<String, SafeNode>,
}

impl NoiseCache {
    pub fn new() -> Self {
        Self {
            compiled_nodes: HashMap::new(),
        }
    }

    pub fn get_or_compile(&mut self, encoded_node: &str) -> &SafeNode {
        // Use entry API to compile only if missing
        if !self.compiled_nodes.contains_key(encoded_node) {
            let node = SafeNode::from_encoded_node_tree(encoded_node)
                .expect("Failed to decode FN2 string");
            self.compiled_nodes.insert(encoded_node.to_string(), node);
        }
        self.compiled_nodes.get(encoded_node).unwrap()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum ChunkState {
    Unknown,
    Air,   // All coarse points <= 0
    Solid, // All coarse points > 0
    Mixed, // Surface/Interpolated
}

pub struct BiomeChunk {
    /// 8x8 grid of Biome IDs (sampled with padding)
    pub biome_grid_2d: [u16; 64],
    /// 8x8x8 grid of Density/Layer data
    pub winning_layer_indices: Box<[u8; 512]>,
    pub coarse_density: Box<[f32; 512]>,
    pub material_ids: Box<[u8; 512]>,
    pub state: ChunkState,
}

impl BiomeChunk {
    pub fn new() -> Self {
        Self {
            biome_grid_2d: [0u16; 64],
            winning_layer_indices: Box::new([0u8; 512]),
            coarse_density: Box::new([0.0f32; 512]),
            material_ids: Box::new([0u8; 512]), // Initialize to empty/0
            state: ChunkState::Air,
        }
    }
}

#[derive(GodotClass)]
#[class(init, base=Resource)]
pub struct VoxelLayer {
    #[export]
    pub layer_name: GString,

    /// The FastNoise2 graph for this specific layer's 3D density
    #[export]
    pub fn2_encoded_data: GString,

    /// Numeric ID for the engine's voxel material palette
    #[export]
    pub material_id: u16, // Updated to u16 per requirements

    /// Curve for controlling the edges of the coarse data sampling. If chunks look jagged use this.w
    #[export]
    pub transition_curve: Option<Gd<Curve>>,

    /// User-supplied noise or texture for detail mapping
    #[export]
    pub detail_source: Option<Gd<Resource>>,

    #[base]
    base: Base<Resource>,
}

#[derive(GodotClass)]
#[class(init, base=Resource)]
pub struct VoxelBiome {
    #[export]
    pub biome_name: GString,

    #[export]
    pub target_temperature: f32,
    #[export]
    pub target_humidity: f32,

    // Wrap the Gd in Option inside the Array for compatibility
    #[export]
    pub layers: Array<Option<Gd<VoxelLayer>>>,

    #[base]
    base: Base<Resource>,
}

#[derive(GodotClass)]
#[class(init, base=Resource)]
pub struct TerrainManagerData {
    /// The global macro-noise graph for Biome distribution (Temp/Humidity)
    #[export]
    pub biome_macro_graph: GString,

    /// List of all possible biomes
    #[export]
    pub registered_biomes: Array<Option<Gd<VoxelBiome>>>,

    #[export]
    pub voxel_dictionary: Option<Gd<VoxelDictionary>>,

    //pub baked_voxel_dictionary: Option<BakedVoxelDictionary>,
    pub baked_registry: BakedRegistry,

    /// Mapping of Chunk coordinates to their processed metadata
    pub chunk_metadata: HashMap<Vector4i, BiomeChunk>,

    pub coarse_cache: HashMap<(Vector3i, u8), ChunkState>,

    #[ignore]
    cache: Option<NoiseCache>,

    #[base]
    base: Base<Resource>,
}

#[godot_api]
impl TerrainManagerData {
    #[func]
    pub fn is_chunk_empty(&self, cx: i32, cy: i32, cz: i32, depth: u8) -> bool {
        let key = (Vector3i::new(cx, cy, cz), depth);
        if let Some(chunk) = self.coarse_cache.get(&key) {
            // Air means the entire 8x8x8 coarse grid was <= 0 density
            return *chunk == ChunkState::Air;
        }
        // If it's not in the cache, we don't know yet.
        // Returning true here might skip chunks incorrectly, so we assume not empty.
        false
    }

    #[func]
    pub fn is_chunk_mixed(&self, cx: i32, cy: i32, cz: i32, depth: u8) -> bool {
        let key = (Vector3i::new(cx, cy, cz), depth);
        if let Some(chunk) = self.coarse_cache.get(&key) {
            // Mixed means it contains the surface (isosurface crosses 0)
            return *chunk == ChunkState::Mixed;
        }
        false
    }

    #[func]
    pub fn is_chunk_solid(&self, cx: i32, cy: i32, cz: i32, depth: u8) -> bool {
        let key = (Vector3i::new(cx, cy, cz), depth);
        if let Some(chunk) = self.coarse_cache.get(&key) {
            return *chunk == ChunkState::Solid;
        }
        false
    }

    #[func]
    pub fn bake_biomes(&mut self) {
        let mut registry = BakedRegistry {
            biomes: HashMap::new(),
            voxel_dictionary: HashMap::new(),
        };

        for (id, biome_opt) in self.registered_biomes.iter_shared().enumerate() {
            if let Some(biome_gd) = biome_opt {
                let biome = biome_gd.bind();
                let mut baked_layers = Vec::new();

                for layer_opt in biome.layers.iter_shared() {
                    if let Some(layer_gd) = layer_opt {
                        let layer = layer_gd.bind();

                        // Bake Curve
                        let mut lut = [0.0f32; 256];
                        if let Some(curve) = &layer.transition_curve {
                            for i in 0..256 {
                                lut[i] = curve.sample(i as f32 / 255.0);
                            }
                        } else {
                            for i in 0..256 {
                                let t = i as f32 / 255.0;
                                lut[i] = t * t * (3.0 - 2.0 * t);
                            }
                        }

                        // Store pure Rust data
                        baked_layers.push(BakedLayer {
                            voxel_id: layer.material_id,
                            curve_lut: lut,
                            fn2_key: layer.fn2_encoded_data.to_string(),
                        });
                    }
                }

                registry.biomes.insert(
                    id as u16,
                    BakedBiome {
                        layers: baked_layers,
                        target_temp: biome.target_temperature,
                        target_hum: biome.target_humidity,
                    },
                );
            }
        }
        self.baked_registry = registry;
        godot_print!("Baked {} biomes.", self.baked_registry.biomes.len());
    }

    #[func]
    pub fn generate_biome_data(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) {
        // Auto-bake if empty
        if self.baked_registry.biomes.is_empty() && !self.registered_biomes.is_empty() {
            self.bake_all();
        }

        let chunk_pos = Vector4i::new(cx, cy, cz, depth);
        let mut chunk = BiomeChunk::new();

        let lod_scale = (1 << depth) as f32;
        let step = 12.8f32 * lod_scale;
        let world_size = 64.0 * lod_scale;
        let x_off = (cx as f32 * world_size) - step;
        let z_off = (cz as f32 * world_size) - step;

        let mut temp_buffer = [0.0f32; 64];
        let mut hum_buffer = [0.0f32; 64];

        let graph_path = self.biome_macro_graph.to_string();
        let macro_node = self.get_cache().get_or_compile(&graph_path).clone();

        macro_node.gen_uniform_grid_2d(&mut temp_buffer, x_off, z_off, 8, 8, step, step, 1337);
        macro_node.gen_uniform_grid_2d(&mut hum_buffer, x_off, z_off, 8, 8, step, step, 4242);

        // Use BakedRegistry for fast lookup (No Binds)
        let registry = &self.baked_registry;
        for i in 0..64 {
            let t = temp_buffer[i];
            let h = hum_buffer[i];

            let mut best_index = 0;
            let mut min_dist_sq = f32::MAX;

            for (id, biome) in &registry.biomes {
                let dt = biome.target_temp - t;
                let dh = biome.target_hum - h;
                let dist_sq = dt * dt + dh * dh;

                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_index = *id;
                }
            }
            chunk.biome_grid_2d[i] = best_index;
        }
        self.chunk_metadata.insert(chunk_pos, chunk);
    }

    fn calculate_closest_biome(&self, temp: f32, humid: f32) -> u16 {
        let mut best_index = 0;
        let mut min_dist_sq = f32::MAX;

        for (i, biome_opt) in self.registered_biomes.iter_shared().enumerate() {
            if let Some(biome_gd) = biome_opt {
                let b = biome_gd.bind();
                let dt = b.target_temperature - temp;
                let dh = b.target_humidity - humid;
                let dist_sq = dt * dt + dh * dh;

                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_index = i as u16;
                }
            }
        }
        best_index
    }

    #[func]
    pub fn generate_density_data(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) {
        let chunk_pos = Vector4i::new(cx, cy, cz, depth);

        // Fetch biome grid
        let biome_grid = match self.chunk_metadata.get(&chunk_pos) {
            Some(c) => c.biome_grid_2d.clone(), // Clone the box to read it
            None => return,
        };

        let lod_scale = (1 << depth) as f32;
        let step = 12.8f32 * lod_scale;
        let world_size = 64.0 * lod_scale;

        let x_start = (cx as f32 * world_size) - step;
        let y_start = (cy as f32 * world_size) - step;
        let z_start = (cz as f32 * world_size) - step;

        let mut unique_layer_nodes: HashMap<String, Box<[f32; 512]>> = HashMap::new();

        // 1a. FIRST PASS: Collect all unique layer keys
        let mut layer_keys_to_compile = Vec::new();
        for &biome_idx in biome_grid.iter() {
            if let Some(biome) = self.baked_registry.biomes.get(&biome_idx) {
                for layer in &biome.layers {
                    if !unique_layer_nodes.contains_key(&layer.fn2_key)
                        && !layer_keys_to_compile.contains(&layer.fn2_key)
                    {
                        layer_keys_to_compile.push(layer.fn2_key.clone());
                    }
                }
            }
        }

        // 1b. SECOND PASS: Compile nodes and generate grids
        for key in layer_keys_to_compile {
            let mut buffer = Box::new([0.0f32; 512]);

            // Now we can safely borrow self mutably
            let node = self.get_cache().get_or_compile(&key).clone();

            // Use Native Order (X, Y, Z) for generation
            node.gen_uniform_grid_3d(
                buffer.as_mut_slice(),
                x_start,
                y_start,
                z_start,
                8,
                8,
                8,
                step,
                step,
                step,
                1337,
            );
            unique_layer_nodes.insert(key, buffer);
        }

        let mut coarse_density = [0.0f32; 512];
        let mut winning_layer_indices = [0u8; 512];
        let mut material_ids = [0u8; 512];

        // 2. ASSIGNMENT LOOP (Pure Rust - No Binds)
        for z in 0..8 {
            for x_coord in 0..8 {
                let biome_idx = biome_grid[z * 8 + x_coord];

                // Get Baked Biome (No Bind)
                if let Some(biome) = self.baked_registry.biomes.get(&biome_idx) {
                    for y_coord in 0..8 {
                        // Engine: [z][x][y] (Y-fast)
                        let engine_idx = (z * 64) + (x_coord * 8) + y_coord;
                        // Noise: [z][y][x] (X-fast)
                        let noise_idx = (z * 64) + (y_coord * 8) + x_coord;

                        let mut max_d = -1.0;
                        let mut win_l = 0;
                        let mut win_mat = 0;

                        for (l_idx, layer) in biome.layers.iter().enumerate() {
                            if let Some(buffer) = unique_layer_nodes.get(&layer.fn2_key) {
                                let val = buffer[noise_idx];
                                if val > max_d {
                                    max_d = val;
                                    win_l = l_idx as u8;
                                    win_mat = layer.voxel_id as u8;
                                }
                            }
                        }
                        coarse_density[engine_idx] = max_d;
                        winning_layer_indices[engine_idx] = win_l;
                        material_ids[engine_idx] = win_mat;
                    }
                }
            }
        }

        // 3. Update Chunk
        if let Some(chunk) = self.chunk_metadata.get_mut(&chunk_pos) {
            chunk.coarse_density = Box::new(coarse_density);
            chunk.winning_layer_indices = Box::new(winning_layer_indices);
            chunk.material_ids = Box::new(material_ids);

            let mut air = true;
            let mut solid = true;
            // Check core 6x6x6
            for z in 1..7 {
                for x in 1..7 {
                    for y in 1..7 {
                        let d = chunk.coarse_density[(z * 64) + (x * 8) + y];
                        if d > 0.0 {
                            air = false;
                        } else {
                            solid = false;
                        }
                        if !air && !solid {
                            break;
                        }
                    }
                    if !air && !solid {
                        break;
                    }
                }
                if !air && !solid {
                    break;
                }
            }
            chunk.state = if solid {
                ChunkState::Solid
            } else if air {
                ChunkState::Air
            } else {
                ChunkState::Mixed
            };
        }
    }

    #[func]
    pub fn clear_coarse_data(&mut self, cx: i32, cy: i32, cz: i32, depth: i32) {
        self.chunk_metadata
            .remove(&Vector4i::new(cx, cy, cz, depth));
    }

    #[func]
    pub fn bake_all(&mut self) {
        self.bake_biomes();

        if let Some(dict_gd) = &self.voxel_dictionary {
            let dict = dict_gd.bind();
            let mut baked_dict = HashMap::new();

            for (id, voxel_opt) in dict.voxel_types.iter_shared().enumerate() {
                if let Some(voxel_gd) = voxel_opt {
                    let v = voxel_gd.bind();
                    // Use the array index as the material_id (0-255)
                    let mat_id = id as u16;

                    baked_dict.insert(mat_id, Self::bake_voxel_type(&v));
                }
            }
            self.baked_registry.voxel_dictionary = baked_dict;
            godot_print!(
                "Baked {} voxel types successfully.",
                self.baked_registry.voxel_dictionary.len()
            );
        }
    }

    // fn bake_voxel_type(voxel: &VoxelType) -> BakedVoxelType {
    //     // 1. Allocate u32 buffer (256x256)
    //     let mut baked_data: Box<[u32; 65536]> = vec![0u32; 65536]
    //         .into_boxed_slice()
    //         .try_into()
    //         .expect("Size mismatch in baked data allocation");

    //     let get_raw_image_data =
    //         |res: &Option<Gd<Resource>>| -> Option<(PackedByteArray, i32, i32)> {
    //             res.as_ref().and_then(|r| {
    //                 let mut img = if let Ok(tex) = r.clone().try_cast::<Texture2D>() {
    //                     tex.get_image()?
    //                 } else if let Ok(image) = r.clone().try_cast::<Image>() {
    //                     image
    //                 } else {
    //                     return None;
    //                 };

    //                 if img.is_compressed() {
    //                     let _ = img.decompress();
    //                 }

    //                 // --- DOWNSAMPLE LOGIC ---
    //                 if img.get_width() > 256 || img.get_height() > 256 {
    //                     img.resize(256, 256);
    //                 }

    //                 img.convert(godot::classes::image::Format::RGBA8);

    //                 let w = img.get_width();
    //                 let h = img.get_height();
    //                 let data = img.get_data();

    //                 if data.is_empty() {
    //                     None
    //                 } else {
    //                     Some((data, w, h))
    //                 }
    //             })
    //         };

    //     let color_info = get_raw_image_data(&voxel.color_texture);
    //     let metal_info = get_raw_image_data(&voxel.metallic_texture);
    //     let emit_info = get_raw_image_data(&voxel.emission_texture);
    //     let disp_info = get_raw_image_data(&voxel.displacement_texture); // New displacement fetch

    //     for y in 0..256 {
    //         for x in 0..256 {
    //             let sample_tiled = |info: &Option<(PackedByteArray, i32, i32)>| -> Color {
    //                 if let Some((data, w, h)) = info {
    //                     let tx = x % *w;
    //                     let ty = y % *h;
    //                     let idx = ((ty * *w + tx) * 4) as usize;
    //                     let slice = data.as_slice();
    //                     if idx + 3 < slice.len() {
    //                         return Color::from_rgba8(
    //                             slice[idx],
    //                             slice[idx + 1],
    //                             slice[idx + 2],
    //                             slice[idx + 3],
    //                         );
    //                     }
    //                 }
    //                 Color::WHITE
    //             };

    //             let c = sample_tiled(&color_info);

    //             // Pack color 15b (5 bits per channel)
    //             let color_15b = ((c.r * voxel.color.r * 31.0) as u32) << 10
    //                 | ((c.g * voxel.color.g * 31.0) as u32) << 5
    //                 | (c.b * voxel.color.b * 31.0) as u32;

    //             // Sample Displacement as 8-bit value (0-255).
    //             // 255 = fully raised (keep solid when carving with indent).
    //             //   0 = fully depressed (carve away).
    //             // When no displacement texture is assigned we write 255 so that
    //             // BakedDisplacementIndent treats this material as fully solid —
    //             // the indent has no effect, which is the correct neutral behaviour.
    //             let disp_val = if disp_info.is_some() {
    //                 (sample_tiled(&disp_info).r * 255.0) as u32
    //             } else {
    //                 255 // No texture → fully raised → neutral (not carved)
    //             };

    //             let is_emit = (sample_tiled(&emit_info).r as f32 / 255.0 * voxel.emission) > 0.1;
    //             let is_metal = if metal_info.is_some() {
    //                 sample_tiled(&metal_info).r > 128.0
    //             } else {
    //                 voxel.metal
    //             };

    //             let is_trans = voxel.transperent;
    //             let is_rough = true;

    //             // Pack into the 32-bit GPU layout (Normals removed)
    //             baked_data[(y * 256 + x) as usize] = Self::pack_baked_voxel(
    //                 color_15b, disp_val, is_emit, is_metal, is_trans, is_rough,
    //             );
    //         }
    //     }

    //     BakedVoxelType {
    //         texture: baked_data,
    //     }
    // }

    fn bake_voxel_type(voxel: &VoxelType) -> BakedVoxelType {
        let mut baked_data: Box<[u32; 65536]> = vec![0u32; 65536]
            .into_boxed_slice()
            .try_into()
            .expect("Size mismatch in baked data allocation");

        // Clamp downsample_factor to a sane range (1 = no extra downsampling, 8 = very aggressive)
        let downsample_factor = voxel.downsample_factor.max(1);

        let get_raw_image_data =
            |res: &Option<Gd<Resource>>| -> Option<(PackedByteArray, i32, i32)> {
                res.as_ref().and_then(|r| {
                    let mut img = if let Ok(tex) = r.clone().try_cast::<Texture2D>() {
                        tex.get_image()?
                    } else if let Ok(image) = r.clone().try_cast::<Image>() {
                        image
                    } else {
                        return None;
                    };

                    if img.is_compressed() {
                        let _ = img.decompress();
                    }

                    // First clamp to 256x256 max, then apply the extra downsample factor.
                    // The tiling logic in sample_tiled uses `%` so any size <= 256 will
                    // repeat correctly across the full 256x256 baked atlas.
                    let target_w = (img.get_width().min(256) / downsample_factor).max(1);
                    let target_h = (img.get_height().min(256) / downsample_factor).max(1);

                    if img.get_width() != target_w || img.get_height() != target_h {
                        img.resize(target_w, target_h);
                    }

                    img.convert(godot::classes::image::Format::RGBA8);

                    let w = img.get_width();
                    let h = img.get_height();
                    let data = img.get_data();

                    if data.is_empty() {
                        None
                    } else {
                        Some((data, w, h))
                    }
                })
            };

        let color_info = get_raw_image_data(&voxel.color_texture);
        let metal_info = get_raw_image_data(&voxel.metallic_texture);
        let emit_info = get_raw_image_data(&voxel.emission_texture);
        let disp_info = get_raw_image_data(&voxel.displacement_texture);

        for y in 0..256 {
            for x in 0..256 {
                let sample_tiled = |info: &Option<(PackedByteArray, i32, i32)>| -> Color {
                    if let Some((data, w, h)) = info {
                        let tx = x % *w;
                        let ty = y % *h;
                        let idx = ((ty * *w + tx) * 4) as usize;
                        let slice = data.as_slice();
                        if idx + 3 < slice.len() {
                            return Color::from_rgba8(
                                slice[idx],
                                slice[idx + 1],
                                slice[idx + 2],
                                slice[idx + 3],
                            );
                        }
                    }
                    Color::WHITE
                };

                let c = sample_tiled(&color_info);

                let color_15b = ((c.r * voxel.color.r * 31.0) as u32) << 10
                    | ((c.g * voxel.color.g * 31.0) as u32) << 5
                    | (c.b * voxel.color.b * 31.0) as u32;

                let disp_val = if disp_info.is_some() {
                    (sample_tiled(&disp_info).r * 255.0) as u32
                } else {
                    255
                };

                let is_emit = (sample_tiled(&emit_info).r as f32 / 255.0 * voxel.emission) > 0.1;
                let is_metal = if metal_info.is_some() {
                    sample_tiled(&metal_info).r > 128.0
                } else {
                    voxel.metal
                };

                let is_trans = voxel.transperent;
                let is_rough = true;

                baked_data[(y * 256 + x) as usize] = Self::pack_baked_voxel(
                    color_15b, disp_val, is_emit, is_metal, is_trans, is_rough,
                );
            }
        }

        BakedVoxelType {
            texture: baked_data,
        }
    }

    fn get_cache(&mut self) -> &mut NoiseCache {
        if self.cache.is_none() {
            self.cache = Some(NoiseCache::new());
        }
        self.cache.as_mut().unwrap()
    }

    pub fn pack_baked_voxel(
        color_15b: u32,
        displacement_8b: u32, // Now a 0-255 value
        emit: bool,
        metal: bool,
        trans: bool,
        rough: bool,
    ) -> u32 {
        let mut packed = color_15b & 0x7FFF; // Bits 0-14

        // Displacement (15-22)
        packed |= (displacement_8b & 0xFF) << 15;

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
}

#[derive(GodotClass)]
#[class(init, base=Resource)]
pub struct VoxelType {
    #[export]
    pub type_name: GString,

    #[export]
    pub color_texture: Option<Gd<Resource>>,

    #[export]
    pub metallic_texture: Option<Gd<Resource>>,

    #[export]
    pub emission_texture: Option<Gd<Resource>>,

    #[export]
    pub displacement_texture: Option<Gd<Resource>>,

    #[export]
    pub color: Color,

    #[export]
    pub emission: f32,

    #[export]
    pub metal: bool,

    #[export]
    pub transperent: bool,

    #[export]
    pub hardness: i32,

    #[export]
    pub downsample_factor: i32,
}

pub struct BakedVoxelType {
    pub texture: Box<[u32; 65536]>,
}

pub struct BakedVoxelDictionary {
    pub baked_voxel_types: Vec<BakedVoxelType>,
}

#[derive(GodotClass)]
#[class(init, base=Resource)]
pub struct VoxelDictionary {
    #[export]
    pub voxel_types: Array<Option<Gd<VoxelType>>>,
}

#[derive(Clone)]
pub struct BakedLayer {
    pub voxel_id: u16,
    pub curve_lut: [f32; 256],
    pub fn2_key: String, // String, not GString
}

#[derive(Clone)]
pub struct BakedBiome {
    pub layers: Vec<BakedLayer>,
    pub target_temp: f32,
    pub target_hum: f32,
}

#[derive(Default)]
pub struct BakedRegistry {
    pub biomes: HashMap<u16, BakedBiome>,
    pub voxel_dictionary: HashMap<u16, BakedVoxelType>,
}
