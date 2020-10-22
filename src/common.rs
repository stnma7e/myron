/// Dim is the type used for spatial dimensions in 3D.
pub type Dim = f32;

pub struct Vec3 {
    x: Dim,
    y: Dim,
    z: Dim,
}

impl Vec3 {
    pub fn zero() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}
