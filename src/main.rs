use std::thread;

use myron::entity::{EntityManager, ComponentManager};
use myron::transform::{TransformManager};
use myron::render::{RenderManager};

fn main() {
    println!("Hello, world!");

    let mut em = EntityManager::new();
    let mut tm = TransformManager::new();
    let mut rm = RenderManager::new();
    let ent = em.create();
    tm.create(ent);

    let compute_thread = thread::spawn(move || {
        loop {
            tm.tick();
        }
    });

    rm.create_vulkan_context();

    compute_thread.join().unwrap();
}
