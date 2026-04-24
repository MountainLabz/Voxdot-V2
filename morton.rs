use mint::Point3;

// Morton code bit interleaving masks.
pub const M01: u128 = 0x4924_9249_2492_4924_9249_2492_4924_9249;
pub const M02: u128 = 0xC30C_30C3_0C30_C30C_30C3_0C30_C30C_30C3;
pub const M04: u128 = 0x0F00_F00F_00F0_0F00_F00F_00F0_0F00_F00F;
pub const M08: u128 = 0xFF00_00FF_0000_FF00_00FF_0000_FF00_00FF;
pub const M16: u128 = 0x0000_FFFF_0000_0000_FFFF_0000_0000_FFFF;
pub const M32: u128 = 0xFFFF_FFFF_0000_0000_0000_0000_FFFF_FFFF;
pub const M64: u128 = 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

/// Converts a morton code into unsigned coordinates.
#[inline(always)]
#[rustfmt::skip]
pub const fn into_point(m: u128) -> Point3<u32> {
    Point3 { x: merge(m, 0), y: merge(m, 1), z: merge(m, 2) }
}

/// Converts unsigned coordinates into a morton code.
#[inline(always)]
pub const fn from_point(p: Point3<u32>) -> u128 {
    space(p.x, 0) | space(p.y, 1) | space(p.z, 2)
}

/// Expands a morton code's domain by `domains`.
#[inline(always)]
pub const fn domain_expand(m: u128, domains: usize) -> u128 {
    m << domains * 3
}

/// Shrinks a morton code's domain by `domains`.
#[inline(always)]
pub const fn domain_shrink(m: u128, domains: usize) -> u128 {
    m >> domains * 3
}

/// Snaps a morton code to the specified `domain`.
#[inline(always)]
pub const fn domain_snap(m: u128, domain: usize) -> u128 {
    let d = domain * 3;
    m >> d << d
}

/// Morton code bit spacer. `0b111` -> `0b1001001`
#[inline(always)]
pub const fn space(i: u32, axis: usize) -> u128 {
    let mut i = i as u128;
    i = (i | (i << 64)) & M32;
    i = (i | (i << 32)) & M16;
    i = (i | (i << 16)) & M08;
    i = (i | (i << 08)) & M04;
    i = (i | (i << 04)) & M02;
    i = (i | (i << 02)) & M01;
    i << axis
}

/// Morton code bit merger. `0b1001001` -> `0b111`
#[inline(always)]
pub const fn merge(mut i: u128, axis: usize) -> u32 {
    i = (i >> axis) & M01;
    i = (i | (i >> 02)) & M02;
    i = (i | (i >> 04)) & M04;
    i = (i | (i >> 08)) & M08;
    i = (i | (i >> 16)) & M16;
    i = (i | (i >> 32)) & M32;
    i = (i | (i >> 64)) & M64;
    i as u32
}
