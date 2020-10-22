pub type EntityId = u64;

#[derive(Clone, Copy)]
pub struct Entity {
    id: EntityId,
}

pub struct EntityManager {
    entities: Vec<Entity>,
    last_used_id: EntityId,
}

pub trait ComponentManager {
    fn create(&mut self, entity: Entity);
    fn tick(&mut self);
}

impl EntityManager {
    pub fn new() -> EntityManager {
        EntityManager {
            entities: vec![],
            last_used_id: 0,
        }
    }

    pub fn create(&mut self) -> Entity {
        let id = self.last_used_id;
        self.last_used_id += 1;

        let ent = Entity { id: id };
        self.entities.push(ent);

        return ent;
    }
}
