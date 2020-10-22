use vulkano::buffer::cpu_pool::{CpuBufferPool, CpuBufferPoolSubbuffer};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
    Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowBuilder};

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};

use std::iter;
use std::sync::Arc;
use std::time::Instant;

use crate::graphics::teapot::{Normal, Vertex, INDICES, NORMALS, VERTICES};

#[derive(Clone)]
pub struct VulkanContext {
    pub event_loop_proxy: EventLoopProxy<VulkanContext>,

    pub surface: Arc<Surface<Window>>,
    pub dimensions: [u32; 2],
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub images: Vec<Arc<SwapchainImage<Window>>>,

    pub uniform_buffer: CpuBufferPool<vs::ty::WorldData>,
    pub render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pub pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    pub framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    pub distance: Point3<f32>,

    pub vs: Arc<vs::Shader>,
    pub fs: Arc<fs::Shader>,
}

#[derive(Clone)]
pub struct Model {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    pub uniform_buffer: CpuBufferPool<vs::ty::ModelData>,
    pub set: Arc<
        PersistentDescriptorSet<(
            (),
            PersistentDescriptorSetBuf<
                CpuBufferPoolSubbuffer<vs::ty::ModelData, Arc<StdMemoryPool>>,
            >,
        )>,
    >,
}

impl VulkanContext {
    pub fn new(event_loop: &EventLoop<VulkanContext>) -> VulkanContext {
        let event_loop_proxy = event_loop.create_proxy();

        let required_extensions = vulkano_win::required_extensions();
        let instance = Box::new(Instance::new(None, &required_extensions, None).unwrap());
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, *instance.clone())
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
        let (swapchain, images) = {
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

        let (pipeline, framebuffers) =
            window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

        let uniform_buffer =
            CpuBufferPool::<vs::ty::WorldData>::new(device.clone(), BufferUsage::all());

        VulkanContext {
            event_loop_proxy: event_loop_proxy,

            surface: surface,
            dimensions: dimensions,
            device: device,
            queue: queue,
            swapchain: swapchain,
            images: images,

            uniform_buffer: uniform_buffer,
            render_pass: render_pass,
            pipeline: pipeline,
            framebuffers: framebuffers,

            distance: Point3::new(0.0, 0.0, 0.0),

            vs: Arc::new(vs),
            fs: Arc::new(fs),
        }
    }

    pub fn new_model(&self, rotation_start: Instant) -> Model {
        let vertices = VERTICES.iter().cloned();
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::all(),
            false,
            vertices,
        )
        .unwrap();

        let normals = NORMALS.iter().cloned();
        let normals_buffer =
            CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::all(), false, normals)
                .unwrap();

        let indices = INDICES.iter().cloned();
        let index_buffer =
            CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::all(), false, indices)
                .unwrap();

        let uniform_buffer =
            CpuBufferPool::<vs::ty::ModelData>::new(self.device.clone(), BufferUsage::all());

        Model::new(
            vertex_buffer.clone(),
            normals_buffer.clone(),
            index_buffer.clone(),
            uniform_buffer,
            rotation_start,
            self.pipeline.clone(),
        )
    }
}

impl Model {
    fn new(
        vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
        normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
        index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
        uniform_buffer: CpuBufferPool<vs::ty::ModelData>,
        rotation_start: Instant,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    ) -> Self {
        let set = Model::create_descriptor_set(
            uniform_buffer.clone(),
            pipeline,
            rotation_start,
            Point3::new(0.0, 0.0, 0.0),
        );
        Model {
            vertex_buffer: vertex_buffer,
            normal_buffer: normal_buffer,
            index_buffer: index_buffer,
            uniform_buffer: uniform_buffer,
            set: set,
        }
    }

    pub fn update(
        &mut self,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        rotation_start: Instant,
        index: usize,
    ) {
        let distance = Point3::new((index as f32) * 0.5, 0.0, 0.0);
        self.set = Model::create_descriptor_set(
            self.uniform_buffer.clone(),
            pipeline,
            rotation_start,
            distance,
        );
    }

    fn create_descriptor_set(
        uniform_buffer: CpuBufferPool<vs::ty::ModelData>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        rotation_start: Instant,
        distance: Point3<f32>,
    ) -> Arc<
        PersistentDescriptorSet<(
            (),
            PersistentDescriptorSetBuf<
                CpuBufferPoolSubbuffer<vs::ty::ModelData, Arc<StdMemoryPool>>,
            >,
        )>,
    > {
        let uniform_buffer_subbuffer = {
            let elapsed = rotation_start.elapsed();
            let rotation =
                elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = Matrix3::from_angle_y(Rad(rotation as f32));
            let scale = Matrix4::from_scale(0.005);

            let translate =
                Matrix4::from_translation(Vector3::new(distance[0], distance[1], distance[2]));

            let uniform_data = vs::ty::ModelData {
                model: (translate * Matrix4::from(rotation) * scale).into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        )
    }
}

pub fn window_size_dependent_setup(
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

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;


layout(set = 0, binding = 0) uniform WorldData {
    mat4 view;
    mat4 proj;
} uniforms;

layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
} model_uniforms;

void main() {
    mat4 worldview = uniforms.view * model_uniforms.model;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
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
