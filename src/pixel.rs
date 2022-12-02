use std::{
	ops::{Add, Mul},
	simd::{Simd, SimdFloat},
};

use glium::texture::PixelValue;

#[derive(Copy, Clone)]
pub struct Pixel(pub Simd<f32, 4>);

unsafe impl PixelValue for Pixel {
	fn get_format() -> glium::texture::ClientFormat {
		glium::texture::ClientFormat::F32F32F32F32
	}
}

impl Pixel {
	pub fn new(r: u8, g: u8, b: u8) -> Self {
		Self(
			Simd::from([r as f32, g as f32, b as f32, 0.0])
				/ Simd::from([255.0, 255.0, 255.0, 1.0]),
		)
	}

	pub fn mul(&self, scalar: f32) -> Self {
		Self(self.0 * Simd::from([scalar, scalar, scalar, 1.0]))
	}

	pub fn set_abs(&mut self) {
		self.0 = self.0.abs();
	}
}

impl Add for Pixel {
	type Output = Pixel;

	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0 + rhs.0)
	}
}

impl Mul for Pixel {
	type Output = Pixel;

	fn mul(self, rhs: Self) -> Self::Output {
		Self(self.0 * rhs.0)
	}
}

impl Mul<Simd<f32, 4>> for &Pixel {
	type Output = Pixel;

	fn mul(self, rhs: Simd<f32, 4>) -> Self::Output {
		Pixel(self.0 * rhs)
	}
}
