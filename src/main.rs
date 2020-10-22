use std::thread;

use myron::entity::{ComponentManager, EntityManager};
use myron::graphics::render::RenderManager;
use myron::transform::TransformManager;

fn main() {
    println!("Hello, world!");

    let mut em = EntityManager::new();
    let mut tm = TransformManager::new();
    let mut rm = RenderManager::new();
    let ent = em.create();
    tm.create(ent);

    let compute_thread = thread::spawn(move || loop {
        tm.tick();
    });

    rm.create_vulkan_context();

    compute_thread.join().unwrap();
}
