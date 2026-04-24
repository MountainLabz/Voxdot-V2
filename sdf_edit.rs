// ===========================================================================
//  sdf_edit.rs  —  SDF Edit System for VoxDot
//
//  Architecture overview
//  ─────────────────────
//  SdfBrush          – pure-data descriptor (shape + operation + material).
//                      Entirely stack-allocated, cheap to clone/pass.
//
//  SdfEditSystem     – owns the "pending edits" queue and exposes
//                      apply_to_terrain(), which stamps a brush into
//                      the octree and marks dirty chunks.
//
//  Displacement indent
//  ───────────────────
//  The baking pipeline (BakedVoxelType) now packs displacement into every
//  u32 in the 256×256 atlas.  Packed u32 layout (from pack_baked_voxel):
//
//    bits  0-14  : color RGB (5-5-5)
//    bits 15-22  : displacement value (8-bit, 0=low/carved, 255=raised/solid)
//    bits 23-26  : (reserved / padding)
//    bit  27     : emit flag
//    bit  28     : metal flag
//    bit  29     : transparent flag
//    bit  30     : rough flag
//
//  During a Subtract stamp, the system samples the baked atlas for the
//  voxel's material at the same tri-planar UV the bake uses.  Texels whose
//  displacement is ABOVE the raise_threshold are kept solid ("raised"),
//  texels below are carved out ("cracks/indents").  Because we read directly
//  from BakedVoxelType, the displacement map is ALWAYS in sync with the
//  visual texture — no separate texture upload needed.
//
//  Extendability
//  ─────────────
//  • Add new shapes     → implement the `SdfShape` trait.
//  • Add new ops        → add a variant to `SdfOp` and one match-arm.
//  • Smoothing          → `SdfOp::SmoothUnion(k)` / `SdfOp::SmoothSub(k)`.
//  • Noise-driven carve → wrap any shape in `Displaced<S>`.
// ===========================================================================

use crate::voxdot_terrain::{ChunkData, VoxelTerrain};
use crate::world_classes::BakedVoxelType;
use godot::prelude::*;

// ---------------------------------------------------------------------------
//  Packed u32 helpers — mirrors pack_baked_voxel() in world_classes.rs
//
//  bits  0-14 : color RGB 5-5-5
//  bits 15-22 : displacement (u8, 0=cavity, 255=raised)
//  bits 23-26 : reserved
//  bit  27    : emit
//  bit  28    : metal
//  bit  29    : transparent
//  bit  30    : rough
// ---------------------------------------------------------------------------

/// Extract the 8-bit displacement value from a packed baked voxel u32.
#[inline(always)]
pub fn unpack_displacement(packed: u32) -> u8 {
    ((packed >> 15) & 0xFF) as u8
}

// ---------------------------------------------------------------------------
//  1. SDF shape primitives
// ---------------------------------------------------------------------------

/// Every shape returns a *signed distance* from a query point.
/// Negative = inside, positive = outside, zero = surface.
pub trait SdfShape: Send + Sync {
    fn signed_distance(&self, p: Vector3) -> f32;
}

/// Axis-aligned sphere.
pub struct Sphere {
    pub center: Vector3,
    pub radius: f32,
}
impl SdfShape for Sphere {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        (p - self.center).length() - self.radius
    }
}

/// Axis-aligned box.
pub struct SdfBox {
    pub center: Vector3,
    pub half_extents: Vector3,
}
impl SdfShape for SdfBox {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let q = (p - self.center).abs() - self.half_extents;
        let q_pos = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        q_pos.length() + q.x.max(q.y).max(q.z).min(0.0)
    }
}

/// Rounded box (chamfered edges).
pub struct RoundBox {
    pub center: Vector3,
    pub half_extents: Vector3,
    pub radius: f32,
}
impl SdfShape for RoundBox {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let q = (p - self.center).abs() - self.half_extents;
        let q_pos = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        q_pos.length() + q.x.max(q.y).max(q.z).min(0.0) - self.radius
    }
}

/// Infinite vertical cylinder  (XZ plane, infinite along Y).
pub struct Cylinder {
    pub center_xz: Vector2, // XZ center
    pub radius: f32,
    pub half_height: f32,
    pub center_y: f32,
}
impl SdfShape for Cylinder {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let d = Vector2::new(
            Vector2::new(p.x, p.z).distance_to(self.center_xz) - self.radius,
            (p.y - self.center_y).abs() - self.half_height,
        );
        d.x.max(d.y).min(0.0) + Vector2::new(d.x.max(0.0), d.y.max(0.0)).length()
    }
}

/// Torus.
pub struct Torus {
    pub center: Vector3,
    pub major_radius: f32, // ring radius
    pub minor_radius: f32, // tube radius
}
impl SdfShape for Torus {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = p - self.center;
        let q = Vector2::new(Vector2::new(p.x, p.z).length() - self.major_radius, p.y);
        q.length() - self.minor_radius
    }
}

/// Vertical Cone.
pub struct Cone {
    pub center: Vector3,
    pub angle_rad: f32, // Angle of the cone
    pub height: f32,
}

impl SdfShape for Cone {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = p - self.center;
        let q = Vector2::new(Vector2::new(p.x, p.z).length(), p.y);
        let c = Vector2::new(self.angle_rad.sin(), self.angle_rad.cos());
        let w = Vector2::new(q.x, q.y - self.height);
        let v = Vector2::new(c.x, -c.y) * q.x.max(c.y * q.x + c.x * q.y);
        let a = (w - v * (w.dot(v) / v.length_squared()).clamp(0.0, 1.0)).length();
        let b = (w - Vector2::new(c.x, 0.0) * w.x.clamp(0.0, c.x)).length();
        a.min(b)
            * (if q.y > 0.0 && q.x < c.x * q.y / c.y {
                -1.0
            } else {
                1.0
            })
    }
}

/// Box Frame (hollow box edges).
pub struct BoxFrame {
    pub center: Vector3,
    pub bounds: Vector3,
    pub thickness: f32,
}

impl SdfShape for BoxFrame {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = (p - self.center).abs() - self.bounds;
        let q = (p + Vector3::new(self.thickness, self.thickness, self.thickness)).abs()
            - Vector3::new(self.thickness, self.thickness, self.thickness);

        let f = |a: f32, b: f32, c: f32| {
            Vector3::new(a.max(0.0), b.max(0.0), c.max(0.0)).length() + a.max(b).max(c).min(0.0)
        };

        f(p.x, q.y, q.z).min(f(q.x, p.y, q.z)).min(f(q.x, q.y, p.z))
    }
}

/// Square Pyramid.
pub struct Pyramid {
    pub center: Vector3,
    pub height: f32,
}

impl SdfShape for Pyramid {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = p - self.center;
        let m2 = self.height * self.height + 0.25;

        let mut px = p.x.abs();
        let mut pz = p.z.abs();
        if pz > px {
            std::mem::swap(&mut px, &mut pz);
        }
        px -= 0.5;
        pz -= 0.5;

        let q = Vector3::new(
            pz,
            self.height * p.y - 0.5 * px,
            self.height * px + 0.5 * p.y,
        );

        let s = (-q.x).max(0.0);
        let t = ((q.y - 0.5 * pz) / (m2 + 0.25)).clamp(0.0, 1.0);

        let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
        let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

        let d2 = if q.y < -0.5 * q.x * m2 && q.y > q.x * m2 {
            0.0
        } else {
            a.min(b)
        };

        ((d2 + q.z * q.z) / m2).sqrt() * (if q.z > 0.0 { 1.0 } else { -1.0 })
    }
}

/// Octahedron (bound).
pub struct Octahedron {
    pub center: Vector3,
    pub s: f32,
}

impl SdfShape for Octahedron {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = (p - self.center).abs();
        let m = p.x + p.y + p.z - self.s;
        let mut q: Vector3;
        if 3.0 * p.x < m {
            q = p;
        } else if 3.0 * p.y < m {
            q = Vector3::new(p.y, p.z, p.x);
        } else if 3.0 * p.z < m {
            q = Vector3::new(p.z, p.x, p.y);
        } else {
            return m * 0.57735027;
        }

        let k = (0.5 * (q.z - q.y + self.s)).clamp(0.0, self.s);
        Vector3::new(q.x, q.y - self.s + k, q.z - k).length()
    }
}

/// Capsule (Line with radius).
pub struct Capsule {
    pub a: Vector3,
    pub b: Vector3,
    pub radius: f32,
}

impl SdfShape for Capsule {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let h = (pa.dot(ba) / ba.length_squared()).clamp(0.0, 1.0);
        (pa - ba * h).length() - self.radius
    }
}

/// Hollow Cylinder.
pub struct HollowCylinder {
    pub center: Vector3,
    pub radius: f32,
    pub thickness: f32,
    pub height: f32,
}

impl SdfShape for HollowCylinder {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let p = p - self.center;
        let d = Vector2::new(
            (Vector2::new(p.x, p.z).length() - self.radius).abs() - self.thickness,
            p.y.abs() - self.height,
        );
        d.x.max(d.y).min(0.0) + Vector2::new(d.x.max(0.0), d.y.max(0.0)).length()
    }
}

/// Stairs with extrusion.
pub struct Stairs {
    pub center: Vector3,
    pub step_width: f32,
    pub step_height: f32,
    pub count: i32,
}

impl SdfShape for Stairs {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let mut p = p - self.center;
        // Basic stair logic: repeat and offset
        let mut d = f32::MAX;
        for i in 0..self.count {
            let offset = Vector3::new(0.0, i as f32 * self.step_height, i as f32 * self.step_width);
            let q = p - offset;
            // Simple box for each step
            let res = (q.abs()
                - Vector3::new(
                    self.step_width * 2.0,
                    self.step_height * 0.5,
                    self.step_width * 0.5,
                ));
            let dist = Vector3::new(res.x.max(0.0), res.y.max(0.0), res.z.max(0.0)).length();
            d = d.min(dist);
        }
        d
    }
}

/// Quad (represented as a very thin oriented box).
pub struct Quad {
    pub center: Vector3,
    pub size: Vector2,
    pub thickness: f32,
}

impl SdfShape for Quad {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let q = (p - self.center).abs();
        let d = Vector3::new(q.x - self.size.x, q.y - self.thickness, q.z - self.size.y);
        Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).length()
            + d.x.max(d.y).max(d.z).min(0.0)
    }
}

pub struct Rotated<S: SdfShape> {
    pub inner: S,
    pub rotation: Basis, // Godot Basis (3x3 matrix)
    pub center: Vector3,
}

impl<S: SdfShape> SdfShape for Rotated<S> {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        // Transform point into local space: p' = R^T * (p - center)
        let local_p = self.rotation.xform_inv(p - self.center);
        self.inner.signed_distance(local_p + self.center)
    }
}

/// A shape that adds world-space fractal noise displacement to any inner shape.
/// Use this to make organic craters, tunnels, etc.
pub struct Displaced<S: SdfShape> {
    pub inner: S,
    pub amplitude: f32,
    pub frequency: f32,
}
impl<S: SdfShape> SdfShape for Displaced<S> {
    #[inline]
    fn signed_distance(&self, p: Vector3) -> f32 {
        let base = self.inner.signed_distance(p);
        // Cheap hash-based noise — no allocations, no external crate needed
        let n = hash_noise(p * self.frequency);
        base + n * self.amplitude
    }
}

/// Inline fast hash noise [-1,1].
#[inline(always)]
fn hash_noise(p: Vector3) -> f32 {
    // Integer hash from Inigo Quilez
    let mut h = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7) as u32;
    h = h.wrapping_mul(0x9e3779b9) ^ (h >> 16);
    h = h.wrapping_mul(0x85ebca6b) ^ (h >> 13);
    h = h.wrapping_mul(0xc2b2ae35) ^ (h >> 16);
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
//  2. SDF operations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfOp {
    /// Fill voxels inside the brush.
    Add,
    /// Remove (carve) voxels inside the brush.
    Subtract,
    /// Only affect voxels that are already solid (paint / replace).
    Replace,
    /// Smooth union — blends edges. `k` = blend radius in world units.
    SmoothUnion(f32),
    /// Smooth subtraction.
    SmoothSubtract(f32),
}

// Smooth-min helper (polynomial, Inigo Quilez).
#[inline]
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

// ---------------------------------------------------------------------------
//  3. Baked displacement indent
// ---------------------------------------------------------------------------

/// Modulates the SDF surface per-voxel using the material's baked displacement atlas.
///
/// The atlas is 256×256. One atlas pixel = one voxel. UVs are computed as:
///   pixel = floor(world_coord / vox_size)  mod 256
/// using the two axes of the dominant face plane from the SDF normal.
/// This gives exactly 1:1 correspondence — each voxel reads one texel,
/// no scaling, no tiling artifacts, matches what the shader renders.
///
/// Displacement offset (world units):
///   atlas 255 → +amplitude  (raise surface outward)
///   atlas 128 → 0           (no change at surface)
///   atlas 0   → -amplitude  (pull surface inward)
///
/// carve_threshold (world units, signed):
///   > 0  whole surface pushed outward + texture modulates on top
///   < 0  whole surface pulled inward  + texture modulates on top
///   = 0  texture centered exactly on the SDF surface
#[derive(Clone)]
pub struct BakedDisplacementIndent {
    pub atlas: std::sync::Arc<Box<[u32; 65536]>>,
    /// Signed world-unit bias applied to the SDF threshold.
    pub carve_threshold: f32,
    /// Maximum displacement distance in world units.
    pub amplitude: f32,
}

impl BakedDisplacementIndent {
    pub fn from_baked(baked: &BakedVoxelType, carve_threshold: f32, amplitude: f32) -> Self {
        Self {
            atlas: std::sync::Arc::new(baked.texture.clone()),
            carve_threshold,
            amplitude,
        }
    }

    /// Returns a signed displacement offset in world units for this voxel.
    ///
    /// `sdf_normal` — the outward normal of the brush SDF at `world_pos`
    ///                (from sdf_gradient). Used to pick the correct atlas plane.
    /// `vox_size`   — size of one voxel in world units. Converts world coords
    ///                to integer atlas pixel coordinates (1 pixel = 1 voxel).
    #[inline]
    pub fn sample_offset(&self, world_pos: Vector3, sdf_normal: Vector3, vox_size: f32) -> f32 {
        let p = world_pos;
        let n = sdf_normal;

        let ax = n.x.abs();
        let ay = n.y.abs();
        let az = n.z.abs();

        // Pick the two axes of the face plane (exclude the dominant normal axis).
        // Pixel coordinate = integer voxel index on that axis, wrapping at 256.
        let (u_coord, v_coord) = if ax >= ay && ax >= az {
            // X-dominant face: use Z and Y
            (p.z, p.y)
        } else if ay >= ax && ay >= az {
            // Y-dominant face (top/bottom): use X and Z
            (p.x, p.z)
        } else {
            // Z-dominant face: use X and Y
            (p.x, p.y)
        };

        let px = ((u_coord / vox_size).floor() as i32).rem_euclid(256) as usize;
        let py = ((v_coord / vox_size).floor() as i32).rem_euclid(256) as usize;

        let raw = unpack_displacement(self.atlas[py * 256 + px]);

        // Remap [0, 255] → [-amplitude, +amplitude]
        // 128 = no displacement, 255 = full outward, 0 = full inward
        (raw as f32 / 127.5 - 1.0) * self.amplitude
    }

    pub fn has_displacement_data(&self) -> bool {
        self.atlas.iter().any(|&p| unpack_displacement(p) > 0)
    }
}

// ---------------------------------------------------------------------------
//  4. The brush — combines shape + operation + material + optional indent
// ---------------------------------------------------------------------------

pub struct SdfBrush {
    /// The shape used for SDF evaluation.
    pub shape: Box<dyn SdfShape>,
    /// What to do when inside the shape.
    pub op: SdfOp,
    /// Voxel material ID to write (only used for Add / Replace / SmoothUnion).
    pub material_id: u32,
    /// Optional baked displacement indent — reads directly from the baked atlas
    /// so cracks/bumps are always in sync with the rendered texture.
    pub displacement_indent: Option<BakedDisplacementIndent>,
    /// World-space AABB of the brush — computed once for chunk culling.
    pub aabb_min: Vector3,
    pub aabb_max: Vector3,
}

impl SdfBrush {
    /// Build a brush, computing a loose AABB from a center + radius bound.
    pub fn new(
        shape: Box<dyn SdfShape>,
        op: SdfOp,
        material_id: u32,
        bounding_radius: f32,
        center: Vector3,
    ) -> Self {
        let pad = Vector3::new(bounding_radius, bounding_radius, bounding_radius);
        Self {
            shape,
            op,
            material_id,
            displacement_indent: None,
            aabb_min: center - pad,
            aabb_max: center + pad,
        }
    }

    /// Attach a baked displacement indent.  The atlas is Arc-shared so this is free.
    pub fn with_displacement(mut self, indent: BakedDisplacementIndent) -> Self {
        self.displacement_indent = Some(indent);
        self
    }
}

// ---------------------------------------------------------------------------
//  5. The edit system
// ---------------------------------------------------------------------------

/// Pending edit — queued until apply_all() is called.
pub struct PendingEdit {
    pub brush: SdfBrush,
}

/// Central manager for SDF edits.
/// Attach this as a component or embed it inside VoxelTerrain.
pub struct SdfEditSystem {
    pub pending: Vec<PendingEdit>,
    pub voxel_size: f32,
}

impl SdfEditSystem {
    pub fn new(voxel_size: f32) -> Self {
        Self {
            pending: Vec::new(),
            voxel_size,
        }
    }

    /// Queue a brush for later batch-apply.
    pub fn enqueue(&mut self, brush: SdfBrush) {
        self.pending.push(PendingEdit { brush });
    }

    /// Apply a single brush immediately into the terrain, returning the list
    /// of (cx,cy,cz,depth) chunk coordinates that were dirtied.
    ///
    /// Call mesh_chunk / texture_chunk_sparse on each dirty coord afterward.
    pub fn apply_brush(
        &self,
        brush: &SdfBrush,
        terrain: &mut VoxelTerrain,
        depth: u8,
    ) -> Vec<(i32, i32, i32, u8)> {
        let chunk_size = 64.0 * self.voxel_size * (1 << depth) as f32;

        // floor/floor for min, floor/floor for max — a chunk owns [n*size, (n+1)*size).
        // Using floor on both ends gives the exact set of chunks whose range overlaps.
        let cx_min = (brush.aabb_min.x / chunk_size).floor() as i32;
        let cy_min = (brush.aabb_min.y / chunk_size).floor() as i32;
        let cz_min = (brush.aabb_min.z / chunk_size).floor() as i32;
        let cx_max = (brush.aabb_max.x / chunk_size).floor() as i32;
        let cy_max = (brush.aabb_max.y / chunk_size).floor() as i32;
        let cz_max = (brush.aabb_max.z / chunk_size).floor() as i32;

        let mut dirty: Vec<(i32, i32, i32, u8)> = Vec::new();
        let vox_size = self.voxel_size * (1 << depth) as f32;

        for cx in cx_min..=cx_max {
            for cy in cy_min..=cy_max {
                for cz in cz_min..=cz_max {
                    if let Some(chunk) = Some(terrain.octree.ensure_chunk_mut(cx, cy, cz, depth)) {
                        if stamp_brush_into_chunk(brush, chunk, cx, cy, cz, vox_size) {
                            dirty.push((cx, cy, cz, depth));
                        }
                    }
                }
            }
        }
        dirty
    }

    /// Drain the queue and apply all pending edits in one pass.
    pub fn apply_all(&mut self, terrain: &mut VoxelTerrain, depth: u8) -> Vec<(i32, i32, i32, u8)> {
        let mut all_dirty: Vec<(i32, i32, i32, u8)> = Vec::new();
        let brushes: Vec<SdfBrush> = self.pending.drain(..).map(|e| e.brush).collect();
        for brush in &brushes {
            let mut dirty = self.apply_brush(brush, terrain, depth);
            all_dirty.append(&mut dirty);
        }
        // Deduplicate dirty chunks so we don't re-mesh the same one twice
        all_dirty.sort_unstable();
        all_dirty.dedup();
        all_dirty
    }
}

// ---------------------------------------------------------------------------
//  6. Core stamping — the hot loop
// ---------------------------------------------------------------------------

/// Pack a voxel visual using the CPU-side format (mirrors VoxelTerrain::pack_visuals).
///
/// Layout: [4-bit empty][4-bit extras][8-bit material][1-bit tex][15-bit color]
///   bits  0-14 : RGB 5-5-5
///   bit   15   : has_tex flag (always true when a material is placed)
///   bits 16-23 : material ID
///   bits 24-27 : extras (emit, metal, trans, rough)
#[inline(always)]
fn pack_visuals_solid(material: u8) -> u32 {
    // White base color (31,31,31 in 5-bit) so the atlas tint is unaffected.
    let color_15b: u32 = (31 << 10) | (31 << 5) | 31;
    // Texture bit always set — we have a material so the shader should sample the atlas.
    let tex_bit: u32 = 1 << 15;
    // Material ID in bits 16-23.
    let mat_part: u32 = (material as u32 & 0xFF) << 16;
    // Default flags: rough=true (bit 3 of extras), rest off.
    let extras_part: u32 = (1u32 << 3) << 24; // rough bit
    color_15b | tex_bit | mat_part | extras_part
}

/// Returns true if any voxel was modified.
///
/// Layout recap (from VoxelTerrain):
///   visuals  index = z * 4096 + y * 64 + x
///   occupancy[col]  col = z * 64 + y,  bit index = x
fn stamp_brush_into_chunk(
    brush: &SdfBrush,
    chunk: &mut ChunkData,
    cx: i32,
    cy: i32,
    cz: i32,
    vox_size: f32,
) -> bool {
    // ── Chunk origin in world space ──────────────────────────────────────────
    let chunk_size = 64.0 * vox_size;
    let origin = Vector3::new(
        cx as f32 * chunk_size,
        cy as f32 * chunk_size,
        cz as f32 * chunk_size,
    );

    // ── Clamp brush AABB to this chunk's voxel range ─────────────────────────
    // Convert brush world AABB into local voxel indices [0,64), clamped.
    // This is the core fix: instead of iterating all 64^3 voxels, we only
    // touch the sub-cube that can possibly be inside the brush.
    let local_min = brush.aabb_min - origin;
    let local_max = brush.aabb_max - origin;

    let x_lo = ((local_min.x / vox_size).floor() as i32).clamp(0, 63) as usize;
    let y_lo = ((local_min.y / vox_size).floor() as i32).clamp(0, 63) as usize;
    let z_lo = ((local_min.z / vox_size).floor() as i32).clamp(0, 63) as usize;
    // +1 because the AABB edge might touch the voxel centre just inside the next cell
    let x_hi = ((local_max.x / vox_size).ceil() as i32 + 1).clamp(0, 64) as usize;
    let y_hi = ((local_max.y / vox_size).ceil() as i32 + 1).clamp(0, 64) as usize;
    let z_hi = ((local_max.z / vox_size).ceil() as i32 + 1).clamp(0, 64) as usize;

    // Early-out if the brush doesn't overlap this chunk at all
    if x_lo >= x_hi || y_lo >= y_hi || z_lo >= z_hi {
        return false;
    }

    // Pre-pack the solid visual for this brush's material (Add/Replace/SmoothUnion).
    let packed_solid = pack_visuals_solid(brush.material_id as u8);

    let mut modified = false;

    for z in z_lo..z_hi {
        for y in y_lo..y_hi {
            let col_idx = z * 64 + y;
            let mut col = chunk.occupancy[col_idx];

            for x in x_lo..x_hi {
                // World-space centre of this voxel
                let wp = origin
                    + Vector3::new(
                        x as f32 * vox_size + vox_size * 0.5,
                        y as f32 * vox_size + vox_size * 0.5,
                        z as f32 * vox_size + vox_size * 0.5,
                    );

                let sdf = brush.shape.signed_distance(wp);
                let vis_idx = z * 4096 + y * 64 + x;
                let bit = 1u64 << x;
                let currently_solid = (col & bit) != 0;

                // Effective SDF threshold, shifted by displacement.
                // Only sample the texture near the surface to avoid affecting
                // the deep interior of the brush.
                let threshold = match brush.displacement_indent {
                    Some(ref indent) => {
                        let surface_band = indent.amplitude * 2.0;
                        if sdf.abs() <= surface_band {
                            let normal = sdf_gradient(&*brush.shape, wp, vox_size * 0.5);
                            indent.carve_threshold + indent.sample_offset(wp, normal, vox_size)
                        } else {
                            indent.carve_threshold
                        }
                    }
                    None => 0.0,
                };

                match brush.op {
                    SdfOp::Add => {
                        if sdf <= threshold {
                            col |= bit;
                            chunk.visuals[vis_idx] = packed_solid;
                            modified = true;
                        }
                    }
                    SdfOp::Subtract => {
                        if sdf <= threshold && currently_solid {
                            col &= !bit;
                            chunk.visuals[vis_idx] = 0;
                            modified = true;
                        }
                    }
                    SdfOp::Replace => {
                        if sdf <= threshold && currently_solid {
                            chunk.visuals[vis_idx] = packed_solid;
                            modified = true;
                        }
                    }
                    SdfOp::SmoothUnion(k) => {
                        let existing = if currently_solid {
                            -vox_size * 0.5
                        } else {
                            vox_size * 0.5
                        };
                        if smin(existing, sdf, k) <= threshold {
                            col |= bit;
                            chunk.visuals[vis_idx] = packed_solid;
                            modified = true;
                        }
                    }
                    SdfOp::SmoothSubtract(k) => {
                        let existing = if currently_solid {
                            -vox_size * 0.5
                        } else {
                            vox_size * 0.5
                        };
                        if smin(existing, -sdf, k) > -threshold && currently_solid {
                            col &= !bit;
                            chunk.visuals[vis_idx] = 0;
                            modified = true;
                        }
                    }
                }
            }

            chunk.occupancy[col_idx] = col;
        }
    }

    if modified {
        chunk.is_dirty = true;
    }
    modified
}

/// Finite-difference SDF gradient — cheap face-normal estimator.
#[inline]
fn sdf_gradient(shape: &dyn SdfShape, p: Vector3, eps: f32) -> Vector3 {
    let dx = shape.signed_distance(p + Vector3::new(eps, 0.0, 0.0))
        - shape.signed_distance(p - Vector3::new(eps, 0.0, 0.0));
    let dy = shape.signed_distance(p + Vector3::new(0.0, eps, 0.0))
        - shape.signed_distance(p - Vector3::new(0.0, eps, 0.0));
    let dz = shape.signed_distance(p + Vector3::new(0.0, 0.0, eps))
        - shape.signed_distance(p - Vector3::new(0.0, 0.0, eps));
    Vector3::new(dx, dy, dz).normalized()
}

// ---------------------------------------------------------------------------
//  7. GDScript-friendly Godot bindings
// ---------------------------------------------------------------------------

/// Exposed to GDScript as a Node so designers can sculpt in-editor or at runtime.
#[derive(GodotClass)]
#[class(base=Node)]
pub struct SdfEditorNode {
    base: Base<Node>,
    system: SdfEditSystem,
    #[export]
    pub voxel_size: f32,
    #[export]
    pub target_depth: u8,
}

#[godot_api]
impl INode for SdfEditorNode {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            system: SdfEditSystem::new(0.1),
            voxel_size: 0.1,
            target_depth: 0,
        }
    }
}

#[godot_api]
impl SdfEditorNode {
    // ── Sphere ──────────────────────────────────────────────────────────────
    #[func]
    pub fn add_sphere(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        material_id: u32,
    ) {
        let brush = SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::Add,
            material_id,
            radius,
            center,
        );
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    #[func]
    pub fn subtract_sphere(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
    ) {
        let brush = SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::Subtract,
            0,
            radius,
            center,
        );
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Box ─────────────────────────────────────────────────────────────────
    #[func]
    pub fn add_box(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        half_extents: Vector3,
        material_id: u32,
    ) {
        let r = half_extents.length();
        let brush = SdfBrush::new(
            Box::new(SdfBox {
                center,
                half_extents,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Indent variants ─────────────────────────────────────────────────────
    //
    // carve_threshold: signed world-unit bias on the SDF surface
    //   > 0  surface pushed outward (additive bumps)
    //   < 0  surface pulled inward  (subtractive cracks)
    //   = 0  texture centered on surface
    //
    // amplitude: max displacement distance in world units
    //   e.g. voxel_size * 1.5 for bumps one-and-a-half voxels deep

    #[func]
    pub fn add_sphere_indent(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let mut brush = SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::Add,
            material_id,
            radius + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    #[func]
    pub fn add_cyllinder(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        half_height: f32,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let mut brush = SdfBrush::new(
            Box::new(Cylinder {
                center_xz: Vector2 {
                    x: center.x,
                    y: center.z,
                },
                radius: radius,
                half_height: half_height,
                center_y: center.y,
            }),
            SdfOp::Add,
            material_id,
            radius + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    #[func]
    pub fn add_box_indent(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        half_extents: Vector3,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = half_extents.length() + amplitude.abs();
        let mut brush = SdfBrush::new(
            Box::new(SdfBox {
                center,
                half_extents,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    #[func]
    pub fn subtract_sphere_indent(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let mut brush = SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::Subtract,
            0,
            radius + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    fn fetch_indent(
        terrain: &Gd<crate::voxdot_terrain::VoxelTerrain>,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) -> Option<BakedDisplacementIndent> {
        let tr = terrain.bind();
        let tm = tr.terrain_manager.as_ref()?.bind();
        let atlas = match tm
            .baked_registry
            .voxel_dictionary
            .get(&(material_id as u16))
        {
            Some(a) => a,
            None => {
                godot_warn!(
                    "SdfEditorNode: material {} not in baked_registry — call bake_all() first.",
                    material_id
                );
                return None;
            }
        };
        let indent = BakedDisplacementIndent::from_baked(atlas, carve_threshold, amplitude);
        if !indent.has_displacement_data() {
            godot_warn!(
                "SdfEditorNode: material {} has no displacement data — assign a \
                 displacement_texture to the VoxelType and call bake_all().",
                material_id
            );
        }
        Some(indent)
    }

    // ── Cone ────────────────────────────────────────────────────────────────
    #[func]
    pub fn add_cone(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        angle_rad: f32,
        height: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = height.max(height * angle_rad.tan()) + amplitude.abs();
        let inner = Cone {
            center,
            angle_rad,
            height,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Box Frame ───────────────────────────────────────────────────────────
    #[func]
    pub fn add_box_frame(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        bounds: Vector3,
        thickness: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = bounds.length() + amplitude.abs();
        let inner = BoxFrame {
            center,
            bounds,
            thickness,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Pyramid ─────────────────────────────────────────────────────────────
    #[func]
    pub fn add_pyramid(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        height: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = (height * height + 0.5).sqrt() + amplitude.abs();
        let inner = Pyramid { center, height };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Octahedron ──────────────────────────────────────────────────────────
    #[func]
    pub fn add_octahedron(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        s: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let inner = Octahedron { center, s };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            s + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Capsule (Line) ──────────────────────────────────────────────────────
    #[func]
    pub fn add_capsule(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        a: Vector3,
        b: Vector3,
        radius: f32,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let center = (a + b) * 0.5;
        let r = (a - b).length() * 0.5 + radius + amplitude.abs();
        let mut brush = SdfBrush::new(
            Box::new(Capsule { a, b, radius }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Hollow Cylinder ─────────────────────────────────────────────────────
    #[func]
    pub fn add_hollow_cylinder(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        thickness: f32,
        height: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = (radius + thickness).max(height) + amplitude.abs();
        let inner = HollowCylinder {
            center,
            radius,
            thickness,
            height,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Stairs ──────────────────────────────────────────────────────────────
    #[func]
    pub fn add_stairs(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        step_width: f32,
        step_height: f32,
        count: i32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let total_h = count as f32 * step_height;
        let total_w = count as f32 * step_width;
        let r = (total_h * total_h + total_w * total_w).sqrt() + amplitude.abs();

        let inner = Stairs {
            center,
            step_width,
            step_height,
            count,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Quad ────────────────────────────────────────────────────────────────
    #[func]
    pub fn add_quad(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        size: Vector2,
        thickness: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let r = size.length() + thickness + amplitude.abs();
        let inner = Quad {
            center,
            size,
            thickness,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            r,
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Updated Cylinder (Existing) ─────────────────────────────────────────
    #[func]
    pub fn add_cylinder(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        half_height: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let inner = Cylinder {
            center_xz: Vector2::new(center.x, center.z),
            radius,
            half_height,
            center_y: center.y,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            radius.max(half_height) + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Torus (Existing) ────────────────────────────────────────────────────
    #[func]
    pub fn add_torus(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        minor_radius: f32,
        major_radius: f32,
        rotation_basis: Basis,
        material_id: u32,
        carve_threshold: f32,
        amplitude: f32,
    ) {
        let indent = Self::fetch_indent(&terrain, material_id, carve_threshold, amplitude);
        let inner = Torus {
            center,
            minor_radius,
            major_radius,
        };
        let mut brush = SdfBrush::new(
            Box::new(Rotated {
                inner,
                rotation: rotation_basis,
                center,
            }),
            SdfOp::Add,
            material_id,
            major_radius + amplitude.abs(),
            center,
        );
        brush.displacement_indent = indent;
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    //////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// ////////////////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////
    /// ////////////////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////
    /// //////////

    // ── Smooth union sphere (blended edges) ──────────────────────────────────
    #[func]
    pub fn smooth_add_sphere(
        &mut self,
        terrain: Gd<crate::voxdot_terrain::VoxelTerrain>,
        center: Vector3,
        radius: f32,
        blend_k: f32,
        material_id: u32,
    ) {
        let brush = SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::SmoothUnion(blend_k),
            material_id,
            radius + blend_k,
            center,
        );
        let mut t = terrain.clone();
        let dirty = self
            .system
            .apply_brush(&brush, &mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Batch: enqueue without immediate remesh ──────────────────────────────
    #[func]
    pub fn enqueue_subtract_sphere(&mut self, center: Vector3, radius: f32) {
        self.system.enqueue(SdfBrush::new(
            Box::new(Sphere { center, radius }),
            SdfOp::Subtract,
            0,
            radius,
            center,
        ));
    }

    #[func]
    pub fn flush_edits(&mut self, terrain: Gd<crate::voxdot_terrain::VoxelTerrain>) {
        let mut t = terrain.clone();
        let dirty = self.system.apply_all(&mut t.bind_mut(), self.target_depth);
        Self::remesh_dirty(&mut t.bind_mut(), dirty);
    }

    // ── Internal: trigger re-mesh on all dirty chunks ────────────────────────
    fn remesh_dirty(
        terrain: &mut crate::voxdot_terrain::VoxelTerrain,
        dirty: Vec<(i32, i32, i32, u8)>,
    ) {
        for (cx, cy, cz, d) in dirty {
            terrain.texture_chunk_sparse(cx, cy, cz, d);
            terrain.update_chunk_texture_sparse(cx, cy, cz, d);
            terrain.mesh_chunk(cx, cy, cz, false, d, true, true);
        }
    }

    #[func]
    pub fn snap_to_grid(&self, position: Vector3, grid_step: Vector3) -> Vector3 {
        let snap_axis = |pos: f32, step: f32| -> f32 {
            if step.abs() < f32::EPSILON {
                pos
            } else {
                (pos / step).round() * step
            }
        };

        Vector3::new(
            snap_axis(position.x, grid_step.x),
            snap_axis(position.y, grid_step.y),
            snap_axis(position.z, grid_step.z),
        )
    }
}
