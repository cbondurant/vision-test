#![feature(iter_array_chunks, portable_simd)]
pub mod pixel;
mod pixel_grid;

use glium::{implement_vertex, uniform};
use pixel::Pixel;
use pixel_grid::PixelGrid;

use std::borrow::Cow;
use std::cell;
use std::error::Error;
use std::simd::{Simd, SimdFloat};
use std::sync::{Arc, Mutex};

use glium::Surface;
use uvc::{Context, Frame};

fn gaussian_blur(grid: &PixelGrid) -> PixelGrid {
	let mut interum = grid.clone();

	let weights = [
		Simd::from([0.10628852, 0.10628852, 0.10628852, 1.0]),
		Simd::from([0.14032134, 0.14032134, 0.14032134, 1.0]),
		Simd::from([0.16577006, 0.16577006, 0.16577006, 1.0]),
		Simd::from([0.17524014, 0.17524014, 0.17524014, 1.0]),
	];

	for (x, y) in grid.coords_in_order() {
		let x = x as i32;
		let y = y as i32;
		*interum.get_mut(x, y) = grid.get(x + 3, y) * weights[0]
			+ grid.get(x + 2, y) * weights[1]
			+ grid.get(x + 1, y) * weights[2]
			+ grid.get(x, y) * weights[3]
			+ grid.get(x - 1, y) * weights[2]
			+ grid.get(x - 2, y) * weights[1]
			+ grid.get(x - 3, y) * weights[0];
	}
	for (x, y) in grid.coords_in_order() {
		let x = x as i32;
		let y = y as i32;
		let pas2px = interum.get_mut(x, y);
		*pas2px = *pas2px
			+ grid.get(x, y + 3) * weights[0]
			+ grid.get(x, y + 2) * weights[1]
			+ grid.get(x, y + 1) * weights[2]
			+ grid.get(x, y) * weights[3]
			+ grid.get(x, y - 1) * weights[2]
			+ grid.get(x, y - 2) * weights[1]
			+ grid.get(x, y - 3) * weights[0];
		*pas2px = pas2px.mul(0.5);
	}
	interum
}

fn bernson_lookup(grid: &PixelGrid, cell_width: u32, sample_distance: u32) -> PixelGrid {
	let horz_cells = grid.height() / cell_width;
	let vert_cells = grid.width() / cell_width;
	let mut cell_container = grid.clone();

	for (x, y) in grid.coords_in_order() {
		let mut min = f32::MAX;
		let mut max = f32::MIN;
		for yy in 0..cell_width / sample_distance {
			for xx in 0..cell_width / sample_distance {
				min = min.min(
					grid.get(
						(x + xx * sample_distance - cell_width / 2) as i32,
						(y + yy * sample_distance - cell_width / 2) as i32,
					)
					.0[0],
				);
				max = max.max(
					grid.get(
						(x + xx * sample_distance - cell_width / 2) as i32,
						(y + yy * sample_distance - cell_width / 2) as i32,
					)
					.0[0],
				)
			}
		}
		let thresh = (min + max) / 2.0;
		*cell_container.get_mut(x as i32, y as i32) =
			Pixel(Simd::from([thresh, thresh, thresh, thresh]));
	}
	cell_container
}

fn frame_to_raw_image(
	frame: &Frame,
) -> Result<glium::texture::RawImage2d<'static, Pixel>, Box<dyn Error>> {
	let mut grid = PixelGrid::pixel_grid_from_frame(frame);

	for pixel in grid.iter_mut() {
		let grey = (pixel.0 * Simd::from([0.299, 0.587, 0.114, 0.0])).reduce_sum();
		pixel.0[0] = grey;
		pixel.0[1] = grey;
		pixel.0[2] = grey;
	}

	let mut sobel = Vec::with_capacity((frame.width() * frame.height()) as usize);

	let gauss = gaussian_blur(&grid);

	let local_window: usize = 10;
	let high_freq = 5;

	let low_threshhold = bernson_lookup(&gauss, 20, 4);
	let local_thresh = bernson_lookup(&gauss, local_window as u32, 2);
	let high_freqthresh = bernson_lookup(&gauss, high_freq, 1);

	let mut last_pixel = Pixel::new(255, 255, 255);
	for (x, y) in grid.coords_in_order() {
		let x = x as i32;
		let y = y as i32;
		// sobel.push((thresholdval * 255.0) as u8);
		// sobel.push((thresholdval * 255.0) as u8);
		// sobel.push((thresholdval * 255.0) as u8);

		let mut pixel = *grid.get(x, y);

		let vlf_diff = pixel.0[0] - low_threshhold.get(x, y).0[0];
		let lf_diff = pixel.0[0] - local_thresh.get(x, y).0[0];
		let hf_diff = pixel.0[0] - high_freqthresh.get(x, y).0[0];
		let global_diff = pixel.0[0] - 0.25;
		if hf_diff.abs() > 0.04 {
			if hf_diff > 0.0 {
				pixel.0 = Simd::from([1.0, 1.0, 1.0, 1.0]);
			} else {
				pixel.0 = Simd::from([0.0, 0.0, 0.0, 1.0]);
			}
		} else if lf_diff.abs() > 0.04 {
			if lf_diff > 0.00 {
				pixel.0 = Simd::from([1.0, 1.0, 1.0, 1.0]);
			} else {
				pixel.0 = Simd::from([0.0, 0.0, 0.0, 1.0]);
			}
		} else if vlf_diff.abs() > 0.04 {
			if vlf_diff > 0.00 {
				pixel.0 = Simd::from([1.0, 1.0, 1.0, 1.0]);
			} else {
				pixel.0 = Simd::from([0.0, 0.0, 0.0, 1.0]);
			}
		} else if global_diff.abs() > 0.1 {
			if global_diff > 0.00 {
				pixel.0 = Simd::from([1.0, 1.0, 1.0, 1.0]);
			} else {
				pixel.0 = Simd::from([0.0, 0.0, 0.0, 1.0]);
			}
		} else {
			pixel.0 = Simd::from([0.5, 0.5, 0.5, 1.0]);
		}

		//pixel.0 /= Simd::from([2.0, 2.0, 2.0, 1.0]);
		sobel.push(pixel);
	}

	// for y in 0..frame.height() as i32 {
	// 	for x in 0..frame.width() as i32 {
	// 		let mut sobel_x = gauss.get(x - 1, y - 1).mul(0.25)
	// 			+ gauss.get(x - 1, y).mul(0.5)
	// 			+ gauss.get(x - 1, y + 1).mul(0.25)
	// 			+ gauss.get(x + 1, y - 1).mul(-0.25)
	// 			+ gauss.get(x + 1, y).mul(-0.5)
	// 			+ gauss.get(x + 1, y + 1).mul(-0.25);

	// 		let mut sobel_y = gauss.get(x - 1, y - 1).mul(0.25)
	// 			+ gauss.get(x, y - 1).mul(0.5)
	// 			+ gauss.get(x + 1, y - 1).mul(0.25)
	// 			+ gauss.get(x - 1, y + 1).mul(-0.25)
	// 			+ gauss.get(x, y + 1).mul(-0.5)
	// 			+ gauss.get(x + 1, y + 1).mul(-0.25);

	// 		sobel_x.set_abs();
	// 		sobel_y.set_abs();
	// 		sobel.push(sobel_x + sobel_y);
	// 	}
	// }
	// println!("calc:  {}", frame.width() * frame.height() * 3);
	// println!("sobel: {}", sobel.len());
	// println!("grid:  {}", grid.make_bytes().len());

	let width = grid.width();
	let height = grid.height();
	let image = glium::texture::RawImage2d {
		data: Cow::Owned(sobel),
		width,
		height,
		format: glium::texture::ClientFormat::F32F32F32F32,
	};
	//::from_raw_rgba(sobel, (new_frame.width(), new_frame.height()));

	Ok(image)
}

fn callback_frame_to_image(
	frame: &Frame,
	data: &mut Arc<Mutex<Option<glium::texture::RawImage2d<Pixel>>>>,
) {
	let image = frame_to_raw_image(frame);
	match image {
		Err(x) => println!("{:#?}", x),
		Ok(x) => {
			let mut data = Mutex::lock(data).unwrap();
			*data = Some(x);
		}
	}
}

fn main() {
	let ctx = Context::new().expect("Could not create context");
	let dev = ctx
		.find_device(None, None, None)
		.expect("Could not find device");

	let description = dev.description().unwrap();
	println!(
		"Found device: Bus {:03} Device {:03} : ID {:04x}:{:04x} {} ({})",
		dev.bus_number(),
		dev.device_address(),
		description.vendor_id,
		description.product_id,
		description.product.unwrap_or_else(|| "Unknown".to_owned()),
		description
			.manufacturer
			.unwrap_or_else(|| "Unknown".to_owned())
	);

	// Open multiple devices by enumerating:
	// let mut list = ctx.devices().expect("Could not get devices");
	// let dev = list.next().expect("No device available");

	let devh = dev.open().expect("Could not open device");

	let format = devh
		.get_preferred_format(|x, y| {
			if x.fps >= y.fps && x.width * x.height >= y.width * y.height {
				x
			} else {
				y
			}
		})
		.unwrap();

	println!("Best format found: {:?}", format);
	let mut streamh = devh.get_stream_handle_with_format(format).unwrap();

	println!(
			"Scanning mode: {:?}\nAuto-exposure mode: {:?}\nAuto-exposure priority: {:?}\nAbsolute exposure: {:?}\nRelative exposure: {:?}\nAboslute focus: {:?}\nRelative focus: {:?}",
			devh.scanning_mode(),
			devh.ae_mode(),
			devh.ae_priority(),
			devh.exposure_abs(),
			devh.exposure_rel(),
			devh.focus_abs(),
			devh.focus_rel(),
		);

	let frame = Arc::new(Mutex::new(None));
	let _stream = streamh
		.start_stream(callback_frame_to_image, frame.clone())
		.unwrap();

	use glium::glutin;
	let events_loop = glutin::event_loop::EventLoop::new();
	let window = glutin::window::WindowBuilder::new().with_title("Mirror");
	let context = glutin::ContextBuilder::new();
	let display = glium::Display::new(window, context, &events_loop).unwrap();

	#[derive(Copy, Clone)]
	pub struct QuadVertex {
		pos: (f32, f32),
	}

	implement_vertex!(QuadVertex, pos);

	let vertices: [QuadVertex; 4] = [
		QuadVertex { pos: (-1.0, -1.0) },
		QuadVertex { pos: (-1.0, 1.0) },
		QuadVertex { pos: (1.0, -1.0) },
		QuadVertex { pos: (1.0, 1.0) },
	];

	let indices: [u8; 6] = [0, 1, 2, 1, 3, 2];

	let vertex_shader_source = r#"
	#version 140
	in vec2 pos;
	out vec2 v_position;
	void main() {
		v_position = (pos + 1.0)/2.0;
		gl_Position = vec4(-pos.x, -pos.y, 0.0, 1.0);
	}
	"#;

	let fragment_shader_source = r#"
	#version 140
	in vec2 v_position;
	out vec4 colour;
	uniform sampler2D u_image;
	void main() {
		vec2 pos = v_position;
		colour = texture(u_image, pos);
	}
	"#;

	let vertices = glium::VertexBuffer::new(&display, &vertices).unwrap();
	let indices = glium::IndexBuffer::new(
		&display,
		glium::index::PrimitiveType::TrianglesList,
		&indices,
	)
	.unwrap();
	let program =
		glium::Program::from_source(&display, vertex_shader_source, fragment_shader_source, None)
			.unwrap();

	let mut buffer: Option<glium::texture::SrgbTexture2d> = None;

	events_loop.run(move |event, _, control_flow| {
		if let glutin::event::Event::WindowEvent {
			event: glutin::event::WindowEvent::CloseRequested,
			..
		} = event
		{
			*control_flow = glutin::event_loop::ControlFlow::Exit;
			return;
		}

		let mut target = display.draw();
		target.clear_color(0.0, 0.0, 1.0, 1.0);

		let mut mutex = Mutex::lock(&frame).unwrap();

		match mutex.take() {
			None => {
				// No new frame to render
			}
			Some(image) => {
				let image = glium::texture::SrgbTexture2d::new(&display, image)
					.expect("Could not use image");
				buffer = Some(image);
			}
		}

		if let Some(ref b) = buffer {
			let uniforms = uniform! { u_image: b };
			target
				.draw(
					&vertices,
					&indices,
					&program,
					&uniforms,
					&Default::default(),
				)
				.unwrap();
		}

		target.finish().unwrap();
	});
}
