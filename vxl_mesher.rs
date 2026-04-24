// NOTES FROM AUTHOR:
// Yeah, so I needed to fit x, y, z, h, w plus normal into 32 bits. 6 bits for x, y, z, h, w each is 30 bits, but normal is 3 bits, so 33 bits. But if I assume that the h and w are going to be at minimum one in length I can use less bits and fit everything
// So for the x, y, z axes positions they're from 0 - 63, but h and w have to be 1 - 64 because a zero calue there mean no heoght or width and thus no quad
// So to make it work I assume that h and w are always -1 length when packing and +1 when unpacking. This way they both use one less bit to pack
// This way I can bit pack the normals, and don't have to put quads in a different array dimension for each normal

/// Fast bitwise downsampling of a 64-bit row into a 32-bit row.
/// A 32-bit voxel is solid if either of its two 64-bit parents are solid.
#[inline(always)]
fn compress_64_to_32(mut v: u64) -> u32 {
    // OR adjacent bits: (bit 0 | bit 1), (bit 2 | bit 3) ...
    v = (v | (v >> 1)) & 0x5555_5555_5555_5555;
    // Standard bit-packing sequence (Inverse of Morton spacing)
    v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v >> 8)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v >> 16)) & 0x0000_0000_FFFF_FFFF;
    v as u32
}

/// Transposes a 32x32 bit matrix (rows are u32).
pub fn transpose_32(m: &mut [u32; 32]) {
    let mut j = 16;
    let mut m_val = 0x0000FFFF;
    while j != 0 {
        for i in (0..32).step_by(2 * j) {
            for k in 0..j {
                let t = (m[i + k + j] ^ (m[i + k] >> j)) & m_val;
                m[i + k] ^= t << j;
                m[i + k + j] ^= t;
            }
        }
        j >>= 1;
        m_val ^= m_val << j;
    }
}

/// High-speed 32^3 Greedy Mesher for Colliders
/// This mirrors the logic of the visual build! macro but is specialized for u32.
pub fn build_collider_fast<F>(chunk: &[u64; 4096], mut callback: F)
where
    F: FnMut(u32, u32, u32, u32, u32, u32), // axis (0-5 matching visual), x, y, z, h, d
{
    // --- 1. DOWNSAMPLE ---
    let mut grid = [0u32; 1024];
    for z in 0..32 {
        let z64 = z * 2;
        for y in 0..32 {
            let y64 = y * 2;
            let combined = chunk[z64 * 64 + y64]
                | chunk[z64 * 64 + y64 + 1]
                | chunk[(z64 + 1) * 64 + y64]
                | chunk[(z64 + 1) * 64 + y64 + 1];
            grid[z * 32 + y] = compress_64_to_32(combined);
        }
    }

    // --- 2. MULTI-AXIS FACE EXTRACTION ---
    let mut mask_data = [0u32; 3072]; // 3 views * 1024

    // XYZ View
    mask_data[0..1024].copy_from_slice(&grid);

    // YZX View
    let mut temp = [0u32; 32];
    for z in 0..32 {
        temp.copy_from_slice(&grid[z * 32..(z + 1) * 32]);
        transpose_32(&mut temp);
        for x in 0..32 {
            mask_data[1024 + x * 32 + z] = temp[x];
        }
    }

    // ZXY View
    for y in 0..32 {
        for z in 0..32 {
            temp[z] = grid[z * 32 + y];
        }
        transpose_32(&mut temp);
        let start = 2048 + y * 32;
        mask_data[start..start + 32].copy_from_slice(&temp);
    }

    // --- 3. CULL & GREEDY MESH ---
    // Process 6 faces matching the visual mesh normals: 0=LEFT, 1=RIGHT, 2=DOWN, 3=UP, 4=FORWARD, 5=BACK
    let mut cull_data = [0u32; 6144]; // 6 faces * 1024

    // For each axis, compute negative and positive faces
    for (axis_idx, neg_face, pos_face) in [(0, 0, 1), (1, 2, 3), (2, 4, 5)] {
        let view_offset = axis_idx * 1024;

        for z in 0..32 {
            for y in 0..32 {
                let row_bits = mask_data[view_offset + z * 32 + y];

                // Negative face (left/down/forward)
                let visible_neg = row_bits & !(row_bits << 1);
                // Positive face (right/up/back)
                let visible_pos = row_bits & !(row_bits >> 1);

                // Store in separate face buffers, transposed for greedy meshing
                for x in 0..32 {
                    if (visible_neg & (1 << x)) != 0 {
                        cull_data[neg_face * 1024 + z * 32 + x] |= 1 << y;
                    }
                    if (visible_pos & (1 << x)) != 0 {
                        cull_data[pos_face * 1024 + z * 32 + x] |= 1 << y;
                    }
                }
            }
        }
    }

    // --- 4. GREEDY MESHING (per face, matching visual mesh) ---
    for face in 0..6 {
        let face_offset = face * 1024;

        for z in 0..32 {
            for x in 0..32 {
                let mut y_bits = cull_data[face_offset + z * 32 + x];

                while y_bits != 0 {
                    let y = y_bits.trailing_zeros();
                    let h = (y_bits >> y).trailing_ones();
                    let mask = if h == 32 { !0 } else { ((1 << h) - 1) << y };

                    y_bits &= !mask;

                    // Greedily extend in Z direction
                    let mut d = 1;
                    while z + d < 32 {
                        let next_bits = cull_data[face_offset + (z + d) * 32 + x];
                        if (next_bits & mask) == mask {
                            cull_data[face_offset + (z + d) * 32 + x] &= !mask;
                            d += 1;
                        } else {
                            break;
                        }
                    }

                    // Emit quad with face index matching visual mesh (0-5)
                    callback(face as u32, x as u32, y, z as u32, h, d as u32);
                }
            }
        }
    }
}

/// Transposes a 64x64 bit matrix effectively.
/// Input: rows are u64s. Output: rows are u64s where bits are swapped across the diagonal.
/// (The x-th bit of the y-th row becomes the y-th bit of the x-th row).
pub fn transpose_64(m: &mut [u64; 64]) {
    let mut j = 32;
    let mut m_val = 0x00000000FFFFFFFF;
    while j != 0 {
        for i in (0..64).step_by(2 * j) {
            for k in 0..j {
                let t = (m[i + k + j] ^ (m[i + k] >> j)) & m_val;
                m[i + k] ^= t << j;
                m[i + k + j] ^= t;
            }
        }
        j >>= 1;
        m_val ^= m_val << j;
    }
}

#[macro_export]
macro_rules! build {
    ($bits:ident, $chunk:expr, $slices:expr, $op:expr) => {{
        // Ensure we are working with u64 for the optimized path
        // (This macro logic assumes 64-bit blocks due to the transpose logic)
        const BITS: usize = 64;
        const LOG2: usize = 6;
        const LOG2_2: usize = 12;

        let chunk_ref: &[u64; 4096] = $chunk;
        let slices_ref: &[[u64; 64]; 6] = $slices;

        // Mask Data Layout:
        // 0..4096: XYZ (Normal view)
        // 4096..8192: YZX (Rotated view: Slice X, Row Z, Bit Y)
        // 8192..12288: ZXY (Rotated view: Slice Y, Row X, Bit Z)
        let mut mask_data = [0u64; 3 << LOG2_2];
        let (xyzi, yzxi, zxyi) = (0, 1 << LOG2_2, 2 << LOG2_2);

        // --- OPTIMIZED VOXEL EXTRACTION (Transposition) ---

        // 1. XYZ View: Direct copy
        // chunk is [z][y] (bits x). We copy straight to mask_data[0]
        mask_data[0..4096].copy_from_slice(chunk_ref);

        // 2. YZX View: We want Slice X, Row Z, Bit Y.
        // We iterate Z. chunk[z] gives us a 64-entry array (rows Y, bits X).
        // If we transpose chunk[z], we get (rows X, bits Y).
        // We then scatter these rows into mask_data so that X is the major index (Slice).
        let mut temp_block = [0u64; 64];
        for z in 0..BITS {
            // Copy chunk[z][0..64] to temp
            let z_offset = z << LOG2;
            temp_block.copy_from_slice(&chunk_ref[z_offset..z_offset + 64]);

            // Transpose: Now temp_block[x] contains bits 'y' for this 'z'
            $crate::vxl_mesher::transpose_64(&mut temp_block);

            // Write to YZX part: mask_data[yzxi + (x * 64) + z]
            for x in 0..BITS {
                mask_data[yzxi + (x << LOG2) + z] = temp_block[x];
            }
        }

        // 3. ZXY View: We want Slice Y, Row X, Bit Z.
        // We iterate Y. We gather chunk[0..64][y].
        // This forms a block where rows are Z, bits are X.
        // Transpose it -> Rows are X, bits are Z.
        // Write to mask_data[zxyi + (y * 64) + x]
        for y in 0..BITS {
            // Gather column Y across all Z
            for z in 0..BITS {
                temp_block[z] = chunk_ref[(z << LOG2) + y];
            }

            // Transpose: Now temp_block[x] contains bits 'z' for this 'y'
            $crate::vxl_mesher::transpose_64(&mut temp_block);

            // Write to ZXY part: mask_data[zxyi + (y * 64) + x]
            // This is actually a direct block copy since mask_data is flat
            let dest_start = zxyi + (y << LOG2);
            mask_data[dest_start..dest_start + 64].copy_from_slice(&temp_block);
        }

        // --- HIDDEN FACE CULLING ---
        let mut cull_data = [0u64; 6 << LOG2_2];
        let (min, max) = (0, BITS - 1);

        for (axis, neg, pos) in [(0, 0, 1), (1, 2, 3), (2, 4, 5)] {
            let (ai, ni, pi) = (axis << LOG2_2, neg << LOG2_2, pos << LOG2_2);

            for z in 0..BITS {
                let zi = z << LOG2;
                let (azi, nzi, pzi) = (ai + zi, ni + zi, pi + zi);

                // Pre-calculate boundary checks if possible, or keep loop tight
                for y in 0..BITS {
                    // Note: In mask_data views, 'y' is the row (0..64), 'x' bits are the column.
                    // The 'z' loop here iterates the Slices of the specific view.

                    let row_bits = mask_data[azi + y];

                    // Branchless optimization for culling
                    // Shift row left/right to find exposed faces
                    let visible_neg = row_bits & !(row_bits << 1);
                    let visible_pos = row_bits & !(row_bits >> 1);

                    // We only do the heavy "neighbor slice" check if we are at the boundary (min/max)
                    // and actually have a face there.

                    // Optimization: Compute boundary masks once
                    let boundary_neg_mask = if slices_ref[neg][z] & (1 << y) != 0 { 1 } else { 0 };
                    let boundary_pos_mask = if slices_ref[pos][z] & (1 << y) != 0 { 1 } else { 0 };

                    // Apply boundary culling:
                    // If bit 0 is set in visible_neg, unset it if boundary neighbor is solid
                    let final_neg = visible_neg & !((boundary_neg_mask) << min);
                    // If bit 63 is set in visible_pos, unset it if boundary neighbor is solid
                    let final_pos = visible_pos & !((boundary_pos_mask) << max);

                    cull_data[nzi + y] = final_neg; // Note: indexing here is [slice][row]
                    cull_data[pzi + y] = final_pos;

                    // Note on indices: In original code:
                    // cull_data[nzi + x] |= 1 << y;
                    // This implies cull_data is transposed relative to mask_data?
                    // Original: mask_data accessed as [azi + y], bits are x.
                    // Original write: cull_data[nzi + x] |= 1 << y.
                    // THIS IS A TRANSPOSE.
                    // The culling step in the original code effectively transposed the result
                    // so the greedy mesher could read rows of "width".

                    // Because we want to keep the greedy mesher fast, we DO need this transpose.
                    // However, doing bit-wise writes ( |= 1 << y ) is slow.
                    // We should buffer the culling results and Transpose them before Greedy phase.
                }
            }
        }

        // --- OPTIMIZED CULLING + TRANSPOSE ---
        // The previous loop attempted to do logic + transpose simultaneously.
        // Let's split it for block-wise speed.

        for (axis, neg, pos) in [(0, 0, 1), (1, 2, 3), (2, 4, 5)] {
            let ai = axis << LOG2_2;
            let ni = neg << LOG2_2;
            let pi = pos << LOG2_2;

            for z in 0..BITS {
                let azi = ai + (z << LOG2);

                // We will collect 64 rows of culled bits, then transpose them
                // so they are ready for the Greedy phase (which expects [z][x] bits y).
                let mut neg_block = [0u64; 64];
                let mut pos_block = [0u64; 64];

                for y in 0..BITS {
                    let row_bits = mask_data[azi + y];

                    // Calculate exposed faces
                    let mut vis_neg = row_bits & !(row_bits << 1);
                    let mut vis_pos = row_bits & !(row_bits >> 1);

                    // Boundary checks
                    if (vis_neg & 1) != 0 {
                         if (slices_ref[neg][z] & (1 << y)) != 0 { vis_neg &= !1; }
                    }
                    if (vis_pos & (1 << 63)) != 0 {
                         if (slices_ref[pos][z] & (1 << y)) != 0 { vis_pos &= !(1 << 63); }
                    }

                    neg_block[y] = vis_neg;
                    pos_block[y] = vis_pos;
                }

                // Transpose the blocks so rows are X and bits are Y (required for Greedy)
                $crate::vxl_mesher::transpose_64(&mut neg_block);
                $crate::vxl_mesher::transpose_64(&mut pos_block);

                // Write to cull_data
                // cull_data is [face][z][x] (bits y)
                let nzi = ni + (z << LOG2);
                let pzi = pi + (z << LOG2);
                cull_data[nzi..nzi+64].copy_from_slice(&neg_block);
                cull_data[pzi..pzi+64].copy_from_slice(&pos_block);
            }
        }

        // Greedy quad extraction
        // (This part is mostly fine, the heavy lifting was in extraction/culling)
        for n in 0..6 {
            let ni = n << LOG2_2;
            for z in 0..BITS {
                let nzi = ni + (z << LOG2);
                for x in 0..BITS {
                    let mut y_bits = cull_data[nzi + x];
                    if y_bits == 0 { continue; } // Quick skip

                    while y_bits != 0 {
                        let y = y_bits.trailing_zeros() as usize;
                        let h = (y_bits >> y).trailing_ones() as usize; // Run length in Y

                        // Construct mask for this run
                        // u64 shift overflow protection: if h=64, 0 >> 0 is 0.
                        // Safe approach:
                        let y_mask = if h == 64 { u64::MAX } else { ((1u64 << h) - 1) << y };

                        y_bits &= !y_mask;

                        let mut d = 1; // Depth (width in the greedy logic)

                        // Check next Z slices to see if we can merge
                        for z_next in z + 1..BITS {
                            let nzi_next = ni + (z_next << LOG2);
                            let y_bits_next = &mut cull_data[nzi_next + x];

                            if (*y_bits_next & y_mask) != y_mask {
                                break;
                            }
                            *y_bits_next &= !y_mask; // Consume bits in next slice
                            d += 1;
                        }

                        $op(n as u32, x as u32, y as u32, z as u32, h as u32, d as u32);
                    }
                }
            }
        }
    }};
}
