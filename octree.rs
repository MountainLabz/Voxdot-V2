use crate::voxdot_terrain::ChunkData;
use godot::prelude::*;
use hashbrown::HashMap;

// Depth 5 = 32^3 chunks per Root (Super-Region)
// Root covers 32 * 64 = 2048 voxels wide
pub const MAX_DEPTH: u8 = 5;
pub const ROOT_SIZE: i32 = 1 << MAX_DEPTH; // 32

#[derive(Clone)]
pub struct OctreeNode {
    pub children: [u32; 8], // u32::MAX if empty
    pub chunk: Option<ChunkData>,
    pub parent: u32, // u32::MAX if root
    pub depth: u8,
}

impl Default for OctreeNode {
    fn default() -> Self {
        Self {
            children: [u32::MAX; 8],
            chunk: None,
            parent: u32::MAX,
            depth: 0,
        }
    }
}

pub struct OctreeManager {
    // Maps Super-Region coordinates (x/32, y/32, z/32) to Root Node Index
    pub roots: HashMap<Vector3i, u32>,
    pub node_pool: Vec<OctreeNode>,
    pub free_list: Vec<u32>,
}

impl OctreeManager {
    pub fn new() -> Self {
        Self {
            roots: HashMap::new(),
            node_pool: Vec::with_capacity(4096), // Pre-allocate for ~4k nodes
            free_list: Vec::new(),
        }
    }

    // --- Core Lifecycle ---

    fn alloc_node(&mut self, depth: u8, parent: u32) -> u32 {
        let node = OctreeNode {
            children: [u32::MAX; 8],
            chunk: None, // Data is attached later
            parent,
            depth,
        };

        if let Some(idx) = self.free_list.pop() {
            self.node_pool[idx as usize] = node;
            idx
        } else {
            let idx = self.node_pool.len() as u32;
            self.node_pool.push(node);
            idx
        }
    }

    /// SAFE ACCESS: Gets a mutable reference to data at any depth.
    pub fn get_chunk_mut(
        &mut self,
        cx: i32,
        cy: i32,
        cz: i32,
        depth: u8,
    ) -> Option<&mut ChunkData> {
        let idx = self.get_or_create_node(cx, cy, cz, depth, false)?;
        self.node_pool[idx as usize].chunk.as_mut()
    }

    /// SAFE CREATION: Finds or creates a node and ensures ChunkData is allocated.
    pub fn ensure_chunk_mut(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) -> &mut ChunkData {
        let idx = self.get_or_create_node(cx, cy, cz, depth, true).unwrap();
        let node = &mut self.node_pool[idx as usize];
        if node.chunk.is_none() {
            node.chunk = Some(ChunkData::new());
        }
        node.chunk.as_mut().unwrap()
    }

    /// EDIT TRAVERSAL: Returns a Leaf (Depth 0) index.
    /// 1. If an LOD chunk (Depth > 0) is found on the path, returns None (Blocked).
    /// 2. If create=true and the path is empty, builds the path to Depth 0 (Air Creation).
    pub fn get_leaf_safe(&mut self, cx: i32, cy: i32, cz: i32, create: bool) -> Option<u32> {
        let root_key = Vector3i::new(cx >> MAX_DEPTH, cy >> MAX_DEPTH, cz >> MAX_DEPTH);
        let mut current_idx = if let Some(&idx) = self.roots.get(&root_key) {
            idx
        } else {
            if !create {
                return None;
            }
            let idx = self.alloc_node(MAX_DEPTH, u32::MAX);
            self.roots.insert(root_key, idx);
            idx
        };

        for d in (1..=MAX_DEPTH).rev() {
            let node = &self.node_pool[current_idx as usize];

            // RULE: If we hit a node with a chunk at depth > 0, we are blocked by an LOD.
            if node.chunk.is_some() && node.depth > 0 {
                return None;
            }

            let bit = d - 1;
            let octant =
                (((cx >> bit) & 1) | (((cy >> bit) & 1) << 1) | (((cz >> bit) & 1) << 2)) as usize;

            let mut child_idx = node.children[octant];
            if child_idx == u32::MAX {
                if !create {
                    return None;
                }
                child_idx = self.alloc_node(d - 1, current_idx);
                self.node_pool[current_idx as usize].children[octant] = child_idx;
            }
            current_idx = child_idx;
        }
        Some(current_idx)
    }

    /// Navigates to a specific node.
    /// If `create` is true, it builds the path.
    /// Returns the index in `node_pool`.
    pub fn get_or_create_node(
        &mut self,
        cx: i32,
        cy: i32,
        cz: i32,
        target_depth: u8,
        create: bool,
    ) -> Option<u32> {
        // 1. Find the Super-Region Root
        // We use bit shifting to find which 32x32x32 area this belongs to.
        let rx = cx >> MAX_DEPTH;
        let ry = cy >> MAX_DEPTH;
        let rz = cz >> MAX_DEPTH;
        let root_key = Vector3i::new(rx, ry, rz);

        let mut current_idx = if let Some(&idx) = self.roots.get(&root_key) {
            idx
        } else {
            if !create {
                return None;
            }
            let idx = self.alloc_node(MAX_DEPTH, u32::MAX);
            self.roots.insert(root_key, idx);
            idx
        };

        // 2. Descend the tree
        // We start at MAX_DEPTH (5) and go down to target_depth
        for d in ((target_depth + 1)..=MAX_DEPTH).rev() {
            // Determine octant for the current level
            // The bit at position (d-1) determines the path
            let bit = d - 1;
            let ox = (cx >> bit) & 1;
            let oy = (cy >> bit) & 1;
            let oz = (cz >> bit) & 1;
            let octant = (ox | (oy << 1) | (oz << 2)) as usize;

            let mut child_idx = self.node_pool[current_idx as usize].children[octant];

            if child_idx == u32::MAX {
                if !create {
                    return None;
                }
                child_idx = self.alloc_node(d - 1, current_idx);
                self.node_pool[current_idx as usize].children[octant] = child_idx;
            }

            current_idx = child_idx;
        }

        Some(current_idx)
    }

    pub fn ensure_chunk(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) -> &mut ChunkData {
        let idx = self.get_or_create_node(cx, cy, cz, depth, true).unwrap();
        let node = &mut self.node_pool[idx as usize];

        if node.chunk.is_none() {
            node.chunk = Some(ChunkData::new());
        }

        node.chunk.as_mut().unwrap()
    }

    /// SAFE DELETE: Removes a chunk at any depth and returns it for cleanup.
    pub fn delete_chunk(&mut self, cx: i32, cy: i32, cz: i32, depth: u8) -> Option<ChunkData> {
        if let Some(idx) = self.get_or_create_node(cx, cy, cz, depth, false) {
            return self.node_pool[idx as usize].chunk.take();
        }
        None
    }

    /// Access a ChunkData at a specific coordinate (Leaf only).
    pub fn get_leaf_data_mut(&mut self, cx: i32, cy: i32, cz: i32) -> Option<&mut ChunkData> {
        let idx = self.get_or_create_node(cx, cy, cz, 0, false)?;
        self.node_pool[idx as usize].chunk.as_mut()
    }

    pub fn get_leaf_data(&self, cx: i32, cy: i32, cz: i32) -> Option<&ChunkData> {
        // Note: We need a non-mut version of get_or_create_node or just clone logic for borrow checker peace
        // For brevity, re-implementing traversal read-only:
        let rx = cx >> MAX_DEPTH;
        let ry = cy >> MAX_DEPTH;
        let rz = cz >> MAX_DEPTH;

        let mut curr = *self.roots.get(&Vector3i::new(rx, ry, rz))?;

        for d in (0..MAX_DEPTH).rev() {
            let octant =
                (((cx >> d) & 1) | (((cy >> d) & 1) << 1) | (((cz >> d) & 1) << 2)) as usize;
            curr = self.node_pool[curr as usize].children[octant];
            if curr == u32::MAX {
                return None;
            }
        }
        self.node_pool[curr as usize].chunk.as_ref()
    }

    // --- LOD System ---

    // NEW: Generic accessor to get chunk data at any depth level
    pub fn get_node_data_mut(
        &mut self,
        cx: i32,
        cy: i32,
        cz: i32,
        depth: u8,
    ) -> Option<&mut ChunkData> {
        let idx = self.get_or_create_node(cx, cy, cz, depth, false)?;
        self.node_pool[idx as usize].chunk.as_mut()
    }

    pub fn sample_visuals(&self, node_idx: u32) -> (Box<[u32; 262_144]>, Box<[u64; 4096]>) {
        let mut buffer = Box::new([0u32; 262_144]);
        let mut occupancy = Box::new([0u64; 4096]);
        let node = &self.node_pool[node_idx as usize];

        for i in 0..8 {
            let child_idx = node.children[i];
            if child_idx == u32::MAX {
                continue;
            }

            let ox = (i & 1) * 32;
            let oy = ((i >> 1) & 1) * 32;
            let oz = ((i >> 2) & 1) * 32;

            self.downsample_child_into(
                buffer.as_mut_slice(),
                occupancy.as_mut_slice(),
                child_idx,
                ox,
                oy,
                oz,
            );
        }
        (buffer, occupancy)
    }

    fn downsample_child_into(
        &self,
        target: &mut [u32],
        occupancy_target: &mut [u64],
        child_idx: u32,
        ox: usize,
        oy: usize,
        oz: usize,
    ) {
        let child = &self.node_pool[child_idx as usize];

        if let Some(data) = &child.chunk {
            // Case A: Child has data (it's a leaf or already an LOD)
            for z in 0..32 {
                for y in 0..32 {
                    let src_row_idx = (oz + z) * 64 + (oy + y);
                    let dst_row_idx = (oz + z) * 64 + (oy + y);

                    for x in 0..32 {
                        // 1. Point Sample Visuals
                        // We sample the first voxel (0,0,0) of every 2x2x2 block in the child
                        let src_x = x * 2;
                        let src_y = y * 2;
                        let src_z = z * 2;

                        let src_vis_idx = src_z * 4096 + src_y * 64 + src_x;
                        let val = data.visuals[src_vis_idx];

                        let dst_vis_idx = (oz + z) * 4096 + (oy + y) * 64 + (ox + x);
                        target[dst_vis_idx] = val;

                        // 2. Point Sample Occupancy
                        // Read the bit from the child's occupancy mask at the SAME (src_x, src_y, src_z)
                        let child_row = data.occupancy[src_z * 64 + src_y];
                        let is_solid = (child_row >> src_x) & 1;

                        if is_solid != 0 {
                            occupancy_target[dst_row_idx] |= 1 << (ox + x);
                        }
                    }
                }
            }
        } else {
            // Case B: Child is an empty internal node; recurse to get its aggregated visuals
            let temp_visuals = self.sample_visuals(child_idx);
            for z in 0..32 {
                for y in 0..32 {
                    let src_row_idx = (oz + z) * 64 + (oy + y);
                    let dst_row_idx = (oz + z) * 64 + (oy + y);
                    for x in 0..32 {
                        // 1. Point Sample Visuals
                        // We sample the first voxel (0,0,0) of every 2x2x2 block in the child
                        let src_x = x * 2;
                        let src_y = y * 2;
                        let src_z = z * 2;

                        let src_vis_idx = src_z * 4096 + src_y * 64 + src_x;
                        let val = temp_visuals.0[src_vis_idx];

                        let dst_vis_idx = (oz + z) * 4096 + (oy + y) * 64 + (ox + x);
                        target[dst_vis_idx] = val;

                        // 2. Point Sample Occupancy
                        // Read the bit from the child's occupancy mask at the SAME (src_x, src_y, src_z)
                        let child_row = temp_visuals.1[src_z * 64 + src_y];
                        let is_solid = (child_row >> src_x) & 1;

                        if is_solid != 0 {
                            occupancy_target[dst_row_idx] |= 1 << (ox + x);
                        }
                    }
                }
            }
        }
    }
}
