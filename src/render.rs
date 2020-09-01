use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass, RenderPass};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError, Surface
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::{Window, WindowBuilder};

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};

use std::iter;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::pin::{Pin};

use crate::teapot::{Normal, Vertex, INDICES, NORMALS, VERTICES};
use crate::entity::{Entity, ComponentManager};

pub struct RenderManager {
    entities: Vec<Entity>,

    event_loop: EventLoop<()>,
    surface: Arc<Surface<Window>>,
    dimensions: [u32; 2],
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,

    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
    rotation_start: Instant,

    vs: vs::Shader,
    fs: fs::Shader
}

impl RenderManager {
    pub fn new() -> RenderManager {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Box::new(Instance::new(None, &required_extensions, None).unwrap());
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, *instance.clone())
            .unwrap();

        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let (mut swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let format = caps.supported_formats[0].0;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                ImageUsage::color_attachment(),
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap()
        };

        let vertices = VERTICES.iter().cloned();
        let vertex_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)
                .unwrap();

        let normals = NORMALS.iter().cloned();
        let normals_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, normals).unwrap();

        let indices = INDICES.iter().cloned();
        let index_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, indices).unwrap();

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            )
            .unwrap(),
        );

        let vs = vs::Shader::load(device.clone()).unwrap();
        let fs = fs::Shader::load(device.clone()).unwrap();

        let (mut pipeline, mut framebuffers) =
            window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
        let recreate_swapchain = false;

        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        let rotation_start = Instant::now();

        RenderManager {
            entities: vec![],

            event_loop: event_loop,
            surface: surface,
            dimensions: dimensions,
            device: device,
            queue: queue,
            swapchain: swapchain,
            images: images,
            vertex_buffer: vertex_buffer,
            normal_buffer: normals_buffer,
            index_buffer: index_buffer,
            uniform_buffer: uniform_buffer,
            render_pass: render_pass,
            pipeline: pipeline,
            framebuffers: framebuffers,



            previous_frame_end: previous_frame_end,
            recreate_swapchain: true,
            rotation_start: rotation_start,

            vs: vs,
            fs: fs,
        }
    }

    pub fn runloop(mut self) -> Self {
        self
    }


    pub fn run(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    self.recreate_swapchain = true;
                }
                Event::RedrawEventsCleared => {
                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if self.recreate_swapchain {
                        let dimensions: [u32; 2] = self.surface.window().inner_size().into();
                        let (new_swapchain, new_images) =
                            match self.swapchain.recreate_with_dimensions(dimensions) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        self.swapchain = new_swapchain;
                        let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                            self.device.clone(),
                            &self.vs,
                            &self.fs,
                            &new_images,
                            self.render_pass.clone(),
                        );
                        self.pipeline = new_pipeline;
                        self.framebuffers = new_framebuffers;
                        self.recreate_swapchain = false;
                    }

                    let uniform_buffer_subbuffer = {
                        let elapsed = self.rotation_start.elapsed();
                        let rotation =
                            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                        let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                        // note: this teapot was meant for OpenGL where the origin is at the lower left
                        //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                        let aspect_ratio = self.dimensions[0] as f32 / self.dimensions[1] as f32;
                        let proj = cgmath::perspective(
                            Rad(std::f32::consts::FRAC_PI_2),
                            aspect_ratio,
                            0.01,
                            100.0,
                        );
                        let view = Matrix4::look_at(
                            Point3::new(0.3, 1.3, 1.0),
                            Point3::new(0.0, 0.0, 0.0),
                            Vector3::new(0.0, -1.0, 0.0),
                        );
                        let scale = Matrix4::from_scale(0.01);

                        let uniform_data = vs::ty::Data {
                            world: Matrix4::from(rotation).into(),
                            view: (view * scale).into(),
                            proj: proj.into(),
                        };

                        self.uniform_buffer.next(uniform_data).unwrap()
                    };

                    let layout = self.pipeline.descriptor_set_layout(0).unwrap();
                    let set = Arc::new(
                        PersistentDescriptorSet::start(layout.clone())
                            .add_buffer(uniform_buffer_subbuffer)
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    let (image_num, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                self.recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    if suboptimal {
                        self.recreate_swapchain = true;
                    }

                    let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                        self.device.clone(),
                        self.queue.family(),
                    )
                    .unwrap();
                    builder
                        .begin_render_pass(
                            self.framebuffers[image_num].clone(),
                            false,
                            vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                        )
                        .unwrap()
                        .draw_indexed(
                            self.pipeline.clone(),
                            &DynamicState::none(),
                            vec![self.vertex_buffer.clone(), self.normal_buffer.clone()],
                            self.index_buffer.clone(),
                            set.clone(),
                            (),
                        )
                        .unwrap()
                        .end_render_pass()
                        .unwrap();
                    let command_buffer = builder.build().unwrap();

                    let future = self.previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            self.previous_frame_end = Some(future.boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            self.recreate_swapchain = true;
                            self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }

    }
}

impl ComponentManager for RenderManager {
    fn create(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    fn tick(&mut self) {
    }
}

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(0.6, 0.0, 0.0);
    vec3 regular_color = vec3(1.0, 0.0, 0.0);

    f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}"
    }
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (
    Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
) {
    let dimensions = images[0].dimensions();

    let depth_buffer =
        AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>();

    // In the triangle example we use a dynamic viewport, as its a simple example.
    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .viewports(iter::once(Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }))
            .fragment_shader(fs.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    (pipeline, framebuffers)
}
