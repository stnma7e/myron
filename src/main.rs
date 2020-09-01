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
    rm.create(ent);

    rm.tick();
}
