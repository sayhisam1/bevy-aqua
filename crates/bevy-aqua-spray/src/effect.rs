//! Shared Hanabi asset and fixed emitter/probe pools.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_hanabi::prelude::*;

use crate::{Emitter, MAX_EMITTERS, MAX_PROBES, Probe};

pub(super) fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut images: ResMut<Assets<Image>>,
) {
    for index in 0..MAX_PROBES {
        commands.spawn((
            Name::new(format!("Aqua spray probe {index}")),
            Probe {
                index,
                cooldown: 0.0,
            },
            Transform::default(),
        ));
    }

    let texture = images.add(spray_texture());
    let effect = effects.add(spray_effect());
    for index in 0..MAX_EMITTERS {
        commands.spawn((
            Name::new(format!("Aqua spray emitter {index}")),
            ParticleEffect::new(effect.clone()),
            EffectProperties::default(),
            EffectMaterial {
                images: vec![texture.clone()],
            },
            Transform::default(),
            Visibility::Hidden,
            Emitter,
        ));
    }
}

fn spray_effect() -> EffectAsset {
    let writer = ExprWriter::new();
    let strength = writer.add_property("strength", 1.0_f32.into());
    let normal = writer.add_property("normal", Vec3::Y.into());
    let current = writer.add_property("current", Vec3::ZERO.into());
    let position = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(0.20).expr(),
        dimension: ShapeDimension::Volume,
    };
    let age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());
    let lifetime = SetAttributeModifier::new(
        Attribute::LIFETIME,
        (writer.lit(0.35) + writer.rand(ScalarType::Float) * writer.lit(0.55)).expr(),
    );
    // Reuse one vec3 draw: flat/no-current keeps the original random samples,
    // upward speed [1.8, 2.8), and world-XZ jitter [-1, 1) per component.
    let random = writer.rand(VectorType::VEC3F);
    let velocity = SetAttributeModifier::new(
        Attribute::VELOCITY,
        (writer.prop(current)
            + (writer.prop(normal) * (writer.lit(1.8) + random.clone().y())
                + random * writer.lit(Vec3::new(2.0, 0.0, 2.0))
                - writer.lit(Vec3::new(1.0, 0.0, 1.0)))
                * writer.prop(strength))
        .expr(),
    );
    let gravity = AccelModifier::new(writer.lit(Vec3::new(0.0, -5.5, 0.0)).expr());
    let drag = LinearDragModifier::new(writer.lit(1.4).expr());
    let texture_slot = writer.lit(0u32).expr();
    let mut color = bevy_hanabi::Gradient::new();
    color.add_key(0.0, Vec4::new(0.75, 0.9, 1.0, 0.9));
    color.add_key(0.45, Vec4::new(0.9, 0.97, 1.0, 0.45));
    color.add_key(1.0, Vec4::new(1.0, 1.0, 1.0, 0.0));
    let mut size = bevy_hanabi::Gradient::new();
    size.add_key(0.0, Vec3::splat(0.24));
    size.add_key(0.35, Vec3::splat(0.60));
    size.add_key(1.0, Vec3::ZERO);
    EffectAsset::new(
        64,
        SpawnerSettings::once(1.0.into()).with_emit_on_start(false),
        {
            let mut module = writer.finish();
            module.add_texture_slot("spray");
            module
        },
    )
    .with_name("Aqua spray")
    .with_simulation_space(SimulationSpace::Global)
    .with_simulation_condition(SimulationCondition::WhenVisible)
    .with_alpha_mode(bevy_hanabi::AlphaMode::Blend)
    .init(position)
    .init(age)
    .init(lifetime)
    .init(velocity)
    .update(gravity)
    .update(drag)
    .render(OrientModifier::new(OrientMode::FaceCameraPosition))
    .render(ColorOverLifetimeModifier::new(color))
    .render(SizeOverLifetimeModifier {
        gradient: size,
        screen_space_size: false,
    })
    .render(ParticleTextureModifier {
        texture_slot,
        sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
    })
}

fn spray_texture() -> Image {
    const SIZE: u32 = 32;
    let mut data = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let position = Vec2::new(x as f32, y as f32) / (SIZE - 1) as f32 * 2.0 - Vec2::ONE;
            let radius_squared = position.length_squared();
            let opacity = (-3.5 * radius_squared).exp() * (1.0 - radius_squared).max(0.0);
            data.push((255.0 * opacity.clamp(0.0, 1.0)).round() as u8);
        }
    }
    Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_init_uses_vec3_properties_and_one_shared_random_draw() {
        let effect = spray_effect();
        assert_eq!(
            effect
                .properties()
                .iter()
                .find(|p| p.name() == "normal")
                .unwrap()
                .default_value(),
            &Vec3::Y.into()
        );
        assert_eq!(
            effect
                .properties()
                .iter()
                .find(|p| p.name() == "current")
                .unwrap()
                .default_value(),
            &Vec3::ZERO.into()
        );
        let mut module = effect.module().clone();
        let property_layout = effect.property_layout();
        let particle_layout = effect.particle_layout();
        let mut shader =
            ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
        // Position, age, lifetime precede the production velocity initializer.
        effect
            .init_modifiers()
            .nth(3)
            .unwrap()
            .apply(&mut module, &mut shader)
            .unwrap();
        assert_eq!(shader.main_code.matches("frand3()").count(), 1);
        assert_eq!(shader.main_code.matches("frand()").count(), 0);
        assert!(
            shader
                .main_code
                .contains("properties[properties_array_index].normal")
        );
        assert!(
            shader
                .main_code
                .contains("properties[properties_array_index].current")
        );
        assert!(
            shader
                .main_code
                .contains("properties[properties_array_index].strength")
        );
    }

    #[test]
    fn spray_texture_has_a_soft_edge() {
        let image = spray_texture();
        let data = image.data.as_deref().expect("CPU texture data");
        let center = data[16 * 32 + 16];
        assert!(center > 240);
        assert_eq!(data[0], 0);
        assert!(data[16 * 32] < center / 8);
    }
}
