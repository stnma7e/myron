use crate::common::Vec3;
use crate::entity::{ComponentManager, Entity, EntityId};

pub struct TransformManager {
    entities: Vec<Entity>,
    locations: Vec<Vec3>,
}

impl TransformManager {
    pub fn new() -> TransformManager {
        TransformManager {
            entities: vec![],
            locations: vec![],
        }
    }

    pub fn transform(&mut self, entity: EntityId) {
        unimplemented!();
    }
}

impl ComponentManager for TransformManager {
    fn create(&mut self, entity: Entity) {
        self.entities.push(entity);
        self.locations.push(Vec3::zero());
    }

    fn tick(&mut self) {}
}
