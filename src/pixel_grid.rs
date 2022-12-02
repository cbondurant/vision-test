use std::slice::IterMut;

use crate::pixel::Pixel;
use uvc::Frame;

#[derive(Clone)]
pub struct PixelGrid {
	width: u32,
	height: u32,
	pub values: Vec<Pixel>,
}

impl PixelGrid {
	pub fn pixel_grid_from_frame(frame: &Frame) -> Self {
		let frame = frame.to_rgb().unwrap();

		let values: Vec<Pixel> = frame
			.to_bytes()
			.iter()
			.array_chunks()
			.map(|v: [&u8; 3]| Pixel::new(*v[0], *v[1], *v[2]))
			.collect();

		assert_eq!(frame.width() * frame.height(), values.len() as u32);
		Self {
			width: frame.width(),
			height: frame.height(),
			values,
		}
	}

	/// Gets a pixel. If outside the real bounds of the grid, it copies the closest real pixel
	#[inline(always)]
	pub fn get(&self, x: i32, y: i32) -> &Pixel {
		let x = x.clamp(0, self.width as i32 - 1) as u32;
		let y = y.clamp(0, self.height as i32 - 1) as u32;

		&self.values[(x + y * self.width) as usize]
	}

	pub fn width(&self) -> u32 {
		self.width
	}

	pub fn height(&self) -> u32 {
		self.height
	}

	/// Gets a pixel. If outside the real bounds of the grid, it copies the closest real pixel
	pub fn get_mut(&mut self, x: i32, y: i32) -> &mut Pixel {
		let x = x.min(self.width as i32 - 1) as u32;
		let y = y.min(self.height as i32 - 1) as u32;

		&mut self.values[(x + y * self.width) as usize]
	}

	pub fn iter_mut(&mut self) -> IterMut<Pixel> {
		self.values.iter_mut()
	}

	pub fn coords_in_order(&self) -> CoordIter {
		CoordIter {
			width: self.width,
			height: self.height,
			current: 0,
		}
	}
}

pub struct CoordIter {
	width: u32,
	height: u32,
	current: u32,
}

impl Iterator for CoordIter {
	type Item = (u32, u32);

	fn next(&mut self) -> Option<Self::Item> {
		if self.current >= self.width * self.height {
			None
		} else {
			let x = self.current % self.width;
			let y = self.current / self.width;
			self.current += 1;
			Some((x, y))
		}
	}
}
