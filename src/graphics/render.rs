use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::swapchain::{self, AcquireError, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use cgmath::{Matrix4, Point3, Rad, Vector3};

use std::sync::Arc;
use std::time::Instant;

use crate::entity::{ComponentManager, Entity};
use crate::graphics::vulkan;
use crate::graphics::vulkan::{window_size_dependent_setup, VulkanContext};

pub struct RenderManager {
    entities: Vec<Entity>,
    context: Option<VulkanContext>,
}

impl RenderManager {
    pub fn new() -> RenderManager {
        RenderManager {
            entities: vec![],
            context: None,
        }
    }

    pub fn create_vulkan_context(&mut self) {
        let event_loop = EventLoop::with_user_event();

        let mut ctx = VulkanContext::new(&event_loop);
        let mut recreate_swapchain = true;

        let rotation_start = vec![
            Instant::now(),
            Instant::now() - std::time::Duration::from_millis(1000),
        ];
        let model = ctx.new_model(rotation_start[0]);
        let model2 = ctx.new_model(rotation_start[1]);
        let mut models = vec![model, model2];

        let mut previous_frame_end = Some(sync::now(ctx.device.clone()).boxed());

        event_loop.run(move |e, _, control_flow| {
            match e {
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
                    recreate_swapchain = true;
                }
                Event::UserEvent(new_context) => {
                    ctx = new_context;
                    recreate_swapchain = true;
                }
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::KeyboardInput { input, .. } => {
                            match input.scancode {
                                0x11 => {
                                    // W
                                    ctx.distance[1] += 0.05
                                }
                                0x1e => {
                                    // A
                                    ctx.distance[0] -= 0.05
                                }
                                0x1f => {
                                    // S
                                    ctx.distance[1] -= 0.05
                                }
                                0x20 => {
                                    // D
                                    ctx.distance[0] += 0.05
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                Event::RedrawEventsCleared => {
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        ctx.dimensions = ctx.surface.window().inner_size().into();
                        let (new_swapchain, new_images) =
                            match ctx.swapchain.recreate_with_dimensions(ctx.dimensions) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        ctx.swapchain = new_swapchain;
                        ctx.images = new_images;
                        let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                            ctx.device.clone(),
                            &ctx.vs,
                            &ctx.fs,
                            &ctx.images,
                            ctx.render_pass.clone(),
                        );
                        ctx.pipeline = new_pipeline;
                        ctx.framebuffers = new_framebuffers;
                        recreate_swapchain = false;
                    }

                    let (image_num, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(ctx.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let uniform_buffer_subbuffer = {
                        let aspect_ratio = ctx.dimensions[0] as f32 / ctx.dimensions[1] as f32;
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

                        let uniform_data = vulkan::vs::ty::WorldData {
                            view: view.into(),
                            proj: proj.into(),
                        };

                        ctx.uniform_buffer.next(uniform_data).unwrap()
                    };

                    let layout = ctx.pipeline.descriptor_set_layout(0).unwrap();
                    let set = Arc::new(
                        PersistentDescriptorSet::start(layout.clone())
                            .add_buffer(uniform_buffer_subbuffer)
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                        ctx.device.clone(),
                        ctx.queue.family(),
                    )
                    .unwrap();

                    for (i, model) in models.iter_mut().enumerate() {
                        model.update(ctx.pipeline.clone(), rotation_start[i], i);
                    }

                    {
                        let mut builder = builder
                            .begin_render_pass(
                                ctx.framebuffers[image_num].clone(),
                                false,
                                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                            )
                            .unwrap();
                        for model in &models {
                            builder = builder
                                .draw_indexed(
                                    ctx.pipeline.clone(),
                                    &DynamicState::none(),
                                    vec![model.vertex_buffer.clone(), model.normal_buffer.clone()],
                                    model.index_buffer.clone(),
                                    (set.clone(), model.set.clone()),
                                    (),
                                )
                                .unwrap();
                        }
                        builder.end_render_pass().unwrap();
                    }

                    let command_buffer = builder.build().unwrap();

                    let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(ctx.queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(ctx.queue.clone(), ctx.swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(ctx.device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            previous_frame_end = Some(sync::now(ctx.device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }
        });
    }
}

impl ComponentManager for RenderManager {
    fn create(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    fn tick(&mut self) {}
}
