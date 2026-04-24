use godot::prelude::*;
pub mod morton;
pub mod octree;
pub mod sdf_edit;
pub mod voxdot_rigidbody;
pub mod voxdot_terrain;
pub mod vxl_mesher;
pub mod world_classes;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}
