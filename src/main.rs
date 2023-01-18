//! The simplest possible example that does something.
// cargo run --release -- -C target=cpu_native
#![allow(clippy::unnecessary_wraps)]

use ggez::graphics::{DrawMode, DrawParam, Mesh, MeshBuilder};
use ggez::input::keyboard::{KeyCode, KeyInput, KeyMods};
use ggez::timer::{self, TimeContext};
use ggez::winit::platform::unix::x11::ffi::WindingRule;
use ggez::{
    event,
    glam::*,
    graphics::{self, Color},
    Context, GameResult,
};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn randparti(boxlen: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let pos = vec![
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(6.27)) as f32,
    ];
    return pos;
}
fn boxNN(key: Vec<i32>, boxlen: f32) -> Vec<Vec<i32>> {
    // box is labeled as [x,y]
    let boxnum = (boxlen) as i32;
    let x = key[0];
    let y = key[1];
    let NNlist = vec![
        vec![
            (x - 1 as i32).rem_euclid(boxnum),
            (y - 1 as i32).rem_euclid(boxnum),
        ],
        vec![
            (x as i32).rem_euclid(boxnum),
            (y - 1 as i32).rem_euclid(boxnum),
        ],
        vec![
            (x + 1 as i32).rem_euclid(boxnum),
            (y - 1 as i32).rem_euclid(boxnum),
        ],
        vec![
            (x - 1 as i32).rem_euclid(boxnum),
            (y as i32).rem_euclid(boxnum),
        ],
        vec![(x as i32).rem_euclid(boxnum), (y as i32).rem_euclid(boxnum)],
        vec![
            (x + 1 as i32).rem_euclid(boxnum),
            (y as i32).rem_euclid(boxnum),
        ],
        vec![
            (x - 1 as i32).rem_euclid(boxnum),
            (y + 1 as i32).rem_euclid(boxnum),
        ],
        vec![
            (x as i32).rem_euclid(boxnum),
            (y + 1 as i32).rem_euclid(boxnum),
        ],
        vec![
            (x + 1 as i32).rem_euclid(boxnum),
            (y + 1 as i32).rem_euclid(boxnum),
        ],
    ];
    return NNlist;
}

fn velandind(
    hashmap: HashMap<Vec<i32>, Vec<usize>>,
    particlelist: &Vec<Vec<f32>>,
    boxlen: f32,
    noise: f32,
) -> Vec<Vec<f32>> {
    let mut outputvec = Vec::new();

    for (key, val) in hashmap.iter() {
        let mut keynoisevec: Vec<f32> = Vec::new();
        //get parti index from surrounding boxes
        let mut indecies_to_check: Vec<usize> = Vec::new();
        for checkingbox in &mut boxNN(key.to_vec(), boxlen).into_iter() {
            if hashmap.contains_key(&checkingbox) {
                indecies_to_check.extend(hashmap.get(&checkingbox).unwrap());
            }
        }

        for activepartiindex in val {
            let mut sum_of_sin: f32 = 0.0;
            let mut sum_of_cos: f32 = 0.0;
            // let mut count = 0;
            let mut rng = rand::thread_rng();

            let mut currentparti = &particlelist[*activepartiindex];
            let mut currentpartix = currentparti[0];
            let mut currentpartiy = currentparti[1];

            for otherindex in &indecies_to_check {
                let mut otherparti = &particlelist[*otherindex];
                let mut otherpartix = otherparti[0];
                let mut otherpartiy = otherparti[1];
                let mut otherpartiangle = otherparti[2];

                let mut dx = (currentpartix - otherpartix).abs();
                let mut dy = (currentpartiy - otherpartiy).abs();

                if dx > boxlen / 2.0 {
                    dx = boxlen - dx;
                }
                if dy > boxlen / 2.0 {
                    dy = boxlen - dy;
                }

                let dist = (dx * dx + dy * dy);

                if dist < 1.0 {
                    // println!("particle within radius for parti {:?}", otherindex);
                    sum_of_sin += otherpartiangle.sin();
                    sum_of_cos += otherpartiangle.cos();
                    // println!("sin  {:?}, cos {:?}", sum_of_sin, sum_of_cos);
                }
            }
            // update the newlist with the calculated velocity this might break at somepoint
            let arctan = f32::atan2((sum_of_sin), sum_of_cos);
            // let randangle = rng.gen_range((-noise / 2.0)..(noise / 2.0)) as f32;
            let mut randangle = 0.0;
            if noise > 0.0 {
                randangle += rng.gen_range((-noise / 2.0)..(noise / 2.0)) as f32;
            }
            let mut updatedangle = arctan + randangle;
            // println!("update angle{:?}", updatedangle);
            // newlist[*activepartiindex][2] = updatedangle;
            let mut newpair = vec![*activepartiindex as f32, updatedangle];
            outputvec.push(newpair)
        }
    }
    return outputvec;
}

const WINSIZE: f32 = 700.0;

struct MainState {
    circle: graphics::Mesh,
    intrad: graphics::Mesh,
    time: i32,
    noise: f32,
    boxlen: f32,
    speed: f32,
    partilist: Vec<Vec<f32>>,
    normvel: f32,
    fps: u32,
    visualiseflag: u32,
    updateflag: u32,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<MainState> {
        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            vec2(0., 0.),
            4.0,
            2.0,
            Color::WHITE,
        )?;

        let mut partinum = 200;
        let boxlen: f32 = 50.0;

        let intrad = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            vec2(0., 0.),
            WINSIZE / boxlen,
            2.0,
            Color::RED,
        )?;
        // generate random list of particles
        let mut parti_list = Vec::new();
        for _ in 0..partinum {
            parti_list.push(randparti(boxlen));
        }

        Ok(MainState {
            circle,
            intrad,
            time: 0,
            noise: 2.0,
            speed: 0.03,
            partilist: parti_list,
            boxlen: boxlen,
            normvel: 1.0,
            fps: 30,
            updateflag: 0,
            visualiseflag: 1,
        })
    }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        // fix frame rate

        let mut DESIRED_FPS: u32 = self.fps;

        use ggez::timer;
        // println!("{:?}", timer::check_update_time(_ctx, DESIRED_FPS));
        while timer::check_update_time(_ctx, DESIRED_FPS) {
            self.updateflag = 1;
            // get all paramater data
            self.time = self.time + 1;
            let k_ctx = &_ctx.keyboard;
            // Increase or decrease speed
            if k_ctx.is_key_pressed(KeyCode::Right) {
                self.speed += 0.01;
                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.speed += 0.09
                }
            }
            if k_ctx.is_key_pressed(KeyCode::Left) {
                self.speed -= 0.01;
                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.speed -= 0.09
                }
                if (self.speed < 0.00) {
                    self.speed = 0.00
                }
            }
            // Increase or decrease noise
            if k_ctx.is_key_pressed(KeyCode::Up) {
                self.noise += 0.1;
                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.noise += 0.5
                }
                if self.noise > 6.1 {
                    self.noise = 6.1
                }
            }
            if k_ctx.is_key_pressed(KeyCode::Down) {
                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.noise -= 0.5
                }
                self.noise -= 0.1;
                if self.noise < 0.0 {
                    self.noise = 0.0
                }
            }

            if k_ctx.is_key_pressed(KeyCode::I) {
                let mut rng = rand::thread_rng();
                let newparti = vec![
                    self.boxlen / 2.0,
                    self.boxlen / 2.0,
                    rng.gen_range((0.0)..(6.27)) as f32,
                ];
                self.partilist.push(newparti);
            }

            if k_ctx.is_key_pressed(KeyCode::Q) {
                let mut rng = rand::thread_rng();
                let newparti = vec![
                    rng.gen_range((0.0)..(self.boxlen)) as f32,
                    rng.gen_range((0.0)..(self.boxlen)) as f32,
                    rng.gen_range((0.0)..(6.27)) as f32,
                ];
                self.partilist.push(newparti);

                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    for _ in 0..9 {
                        let mut rng = rand::thread_rng();
                        let newparti = vec![
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(6.27)) as f32,
                        ];
                        self.partilist.push(newparti);
                    }
                }

                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::CTRL) {
                    for _ in 0..99 {
                        let mut rng = rand::thread_rng();
                        let newparti = vec![
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(6.27)) as f32,
                        ];
                        self.partilist.push(newparti);
                    }
                }
            }

            if k_ctx.is_key_pressed(KeyCode::W) {
                self.partilist.pop();

                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    for _ in 0..9 {
                        self.partilist.pop();
                    }
                }

                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::CTRL) {
                    for _ in 0..99 {
                        self.partilist.pop();
                    }
                }
            }

            if k_ctx.is_key_pressed(KeyCode::V) {
                self.visualiseflag = 1;
            };
            if k_ctx.is_key_pressed(KeyCode::B) {
                self.visualiseflag = 0;
            };
            if k_ctx.is_key_pressed(KeyCode::E) {
                let len = self.partilist.len();
                for _ in 0..len {
                    self.partilist.pop();
                }
                for _ in 0..len {
                    let mut rng = rand::thread_rng();
                    let newparti = vec![
                        rng.gen_range((0.0)..(self.boxlen)) as f32,
                        rng.gen_range((0.0)..(self.boxlen)) as f32,
                        rng.gen_range((0.0)..(6.27)) as f32,
                    ];
                    self.partilist.push(newparti);
                }
            }

            if k_ctx.is_key_pressed(KeyCode::R) {
                for _ in 0..self.partilist.len() {
                    self.partilist.pop();
                }
            }

            if k_ctx.is_key_pressed(KeyCode::A) {
                self.boxlen += 1.0;
                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.boxlen -= 1.0;
                    let density = self.partilist.len() as f32 / (self.boxlen * self.boxlen);
                    self.boxlen += 1.0;
                    let N = (density * self.boxlen * self.boxlen).ceil();

                    for _ in 0..self.partilist.len() {
                        self.partilist.pop();
                    }

                    for _ in 0..(N as i32) {
                        let mut rng = rand::thread_rng();
                        let newparti = vec![
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(6.27)) as f32,
                        ];
                        self.partilist.push(newparti);
                    }
                }
            }

            if k_ctx.is_key_pressed(KeyCode::S) {
                self.boxlen -= 1.0;
                if self.boxlen < 4.0 {
                    self.boxlen = 4.0
                }

                if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                    self.boxlen += 1.0;
                    let density = self.partilist.len() as f32 / (self.boxlen * self.boxlen);
                    self.boxlen -= 1.0;
                    let N = (density * self.boxlen * self.boxlen).ceil();

                    for _ in 0..self.partilist.len() {
                        self.partilist.pop();
                    }

                    for _ in 0..(N as i32) {
                        let mut rng = rand::thread_rng();
                        let newparti = vec![
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(6.27)) as f32,
                        ];
                        self.partilist.push(newparti);
                    }
                }
            }

            if k_ctx.is_key_pressed(KeyCode::Z) {
                self.fps += 1;
            }

            if k_ctx.is_key_pressed(KeyCode::X) {
                self.fps -= 1;
            }

            // calculate Normalised Velocity
            if (self.time % 10) == 0 {
                let mut sumsin: f32 = 0.0;
                let mut sumcos: f32 = 0.0;
                for parti in &self.partilist {
                    sumsin += &parti[2].sin();
                    sumcos += &parti[2].cos();
                }
                self.normvel = (1.0 / self.partilist.len() as f32)
                    * f32::sqrt((sumsin * sumsin) + (sumcos * sumcos));
            }

            // update the position of the particles

            let now = Instant::now();
            for k in 0..(self.partilist.len()) {
                self.partilist[k][0] += self.partilist[k][2].sin() * self.speed;
                self.partilist[k][1] += self.partilist[k][2].cos() * self.speed;
                if self.partilist[k][0] < 0.0 {
                    self.partilist[k][0] += self.boxlen
                };
                if self.partilist[k][0] > self.boxlen {
                    self.partilist[k][0] -= self.boxlen
                };
                if self.partilist[k][1] < 0.0 {
                    self.partilist[k][1] += self.boxlen
                };
                if self.partilist[k][1] > self.boxlen {
                    self.partilist[k][1] -= self.boxlen
                };
            }
            let elapsed = now.elapsed();
            if self.time % 100 == 0 {
                println!("position update: {:.2?}", elapsed);
            }

            /// creating hash mapping
            let now = Instant::now();
            let mut boxhashmap = HashMap::new();
            for parti_indx in 0..self.partilist.len() {
                let mut boxindx = vec![
                    self.partilist[parti_indx as usize][0].floor() as i32,
                    self.partilist[parti_indx as usize][1].floor() as i32,
                ];

                boxhashmap
                    .entry(boxindx)
                    .or_insert(Vec::new())
                    .push(parti_indx);
            }
            let elapsed = now.elapsed();
            if self.time % 100 == 0 {
                println!("hashmap update: {:.2?}", elapsed);
            }
            // all code to go between this

            //Update velocity step
            //Go though each box and update velocity
            // creating a new list to update the velocities

            // let mut newlist: Vec<Vec<f32>> = self.partilist.clone();
            // for (key, val) in boxhashmap.iter() {
            //     //get parti index from surrounding boxes
            //     let mut indecies_to_check: Vec<usize> = Vec::new();
            //     for checkingbox in &mut boxNN(key.to_vec(), self.boxlen).into_iter() {
            //         if boxhashmap.contains_key(&checkingbox) {
            //             indecies_to_check.extend(boxhashmap.get(&checkingbox).unwrap());
            //         }
            //     }

            //     // update the velocity of each particle the box
            //     for activepartiindex in val {
            //         let mut sum_of_sin: f32 = 0.0;
            //         let mut sum_of_cos: f32 = 0.0;
            //         let mut rng = rand::thread_rng();

            //         let mut currentparti = &self.partilist[*activepartiindex];
            //         let mut currentpartix = currentparti[0];
            //         let mut currentpartiy = currentparti[1];

            //         for otherindex in &indecies_to_check {
            //             let mut otherparti = &self.partilist[*otherindex];
            //             let mut otherpartix = otherparti[0];
            //             let mut otherpartiy = otherparti[1];
            //             let mut otherpartiangle = otherparti[2];

            //             let mut dx = (currentpartix - otherpartix).abs();
            //             let mut dy = (currentpartiy - otherpartiy).abs();

            //             if dx > self.boxlen / 2.0 {
            //                 dx = self.boxlen - dx;
            //             }
            //             if dy > self.boxlen / 2.0 {
            //                 dy = self.boxlen - dy;
            //             }

            //             let dist = (dx * dx + dy * dy);

            //             if dist < 1.0 {
            //                 sum_of_sin += otherpartiangle.sin();
            //                 sum_of_cos += otherpartiangle.cos();
            //             }
            //         }
            //         // update the newlist with the calculated velocity this might break at somepoint
            //         let arctan = f32::atan2((sum_of_sin), sum_of_cos);
            //         let mut randangle = 0.0;
            //         if self.noise > 0.0 {
            //             randangle += rng.gen_range((-self.noise / 2.0)..(self.noise / 2.0)) as f32;
            //         }
            //         let mut updatedangle = arctan + randangle;
            //         newlist[*activepartiindex][2] = updatedangle;
            //     }
            // }

            // // updating parti_list from the new list
            // self.partilist = newlist.clone();
            // // all new code is to go above here
            // all new code to go under this

            let now = Instant::now();
            let cpu_count = 8;
            let list: Vec<_> = boxhashmap.into_iter().collect();

            // println!("list {:?}", list);
            let chunk_len = (list.len() / cpu_count) as usize + 1;

            let chunks: Vec<HashMap<_, _>> = list
                .chunks(chunk_len)
                .map(|c| c.iter().cloned().collect())
                .collect();

            let parallellist: Vec<Vec<Vec<f32>>> = chunks
                .par_iter()
                .map(|chunk| velandind(chunk.clone(), &self.partilist, self.boxlen, self.noise))
                .collect();

            let flattenlist = parallellist.into_iter().flatten().collect::<Vec<_>>();
            // println!("flattenlist{:?}", flattenlist);
            for change in flattenlist {
                self.partilist[change[0] as usize][2] = change[1]
            }
            let elapsed = now.elapsed();
            if self.time % 100 == 0 {
                println!("velocity update: {:.2?}", elapsed);
            }
            // all new code to go above this
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        if self.updateflag == 1 {
            self.updateflag = 0;

            let mut canvas =
                graphics::Canvas::from_frame(ctx, graphics::Color::from([0.1, 0.2, 0.3, 1.0]));

            // println!("{:?}", timer::check_update_time(ctx, self.fps));
            // Draw each particle and push to window
            // for particle in 0..(self.partilist.len()) {
            //     canvas.draw(
            //         &self.circle,
            //         Vec2::new(
            //             self.partilist[particle][0] * (WINSIZE / self.boxlen),
            //             (self.boxlen - self.partilist[particle][1]) * (WINSIZE / self.boxlen),
            //         ),
            //     );
            // }

            // create mesh of particles and draw to screen

            // while timer::check_update_time(ctx, self.fps) {
            let now = Instant::now();
            if self.visualiseflag == 1 {
                if self.partilist.len() > 0 {
                    let mut mesh_builder = MeshBuilder::new();
                    for particle in self.partilist.iter() {
                        mesh_builder.circle(
                            DrawMode::fill(),
                            Vec2::new(
                                particle[0] * (WINSIZE / self.boxlen),
                                (self.boxlen - particle[1]) * (WINSIZE / self.boxlen),
                            ),
                            2.0,
                            10.0,
                            Color::WHITE,
                        );
                    }

                    for particle in self.partilist.iter() {
                        mesh_builder.line(
                            &[
                                Vec2::new(
                                    particle[0] * (WINSIZE / self.boxlen),
                                    (self.boxlen - particle[1]) * (WINSIZE / self.boxlen),
                                ),
                                Vec2::new(
                                    (particle[0] + particle[2].sin()) * (WINSIZE / self.boxlen),
                                    (self.boxlen - (particle[1] + particle[2].cos()))
                                        * (WINSIZE / self.boxlen),
                                ),
                            ],
                            1.0,
                            Color::WHITE,
                        );
                    }
                    // let mesh = mesh_builder.build();
                    let mesh = Mesh::from_data(ctx, mesh_builder.build());
                    canvas.draw(&mesh, DrawParam::default());

                    let intrad = graphics::Mesh::new_circle(
                        ctx,
                        graphics::DrawMode::fill(),
                        vec2(0., 0.),
                        WINSIZE / (self.boxlen * 2.0),
                        2.0,
                        Color::RED,
                    )?;

                    let redcirc = graphics::Mesh::new_circle(
                        ctx,
                        graphics::DrawMode::fill(),
                        vec2(0., 0.),
                        4.0,
                        2.0,
                        Color::RED,
                    )?;

                    let redline = graphics::Mesh::new_line(
                        ctx,
                        &[
                            Vec2::new(
                                self.partilist[0][0] * (WINSIZE / self.boxlen),
                                (self.boxlen - (self.partilist[0][1])) * (WINSIZE / self.boxlen),
                            ),
                            Vec2::new(
                                (self.partilist[0][0] + (self.partilist[0][2].sin() * 2.0))
                                    * (WINSIZE / self.boxlen),
                                (self.boxlen
                                    - (self.partilist[0][1] + (2.0 * self.partilist[0][2].cos())))
                                    * (WINSIZE / self.boxlen),
                            ),
                        ],
                        5.0,
                        Color::RED,
                    )?;

                    canvas.draw(&redline, DrawParam::default());

                    canvas.draw(
                        &intrad,
                        Vec2::new(
                            self.partilist[0][0] * (WINSIZE / self.boxlen),
                            (self.boxlen - self.partilist[0][1]) * (WINSIZE / self.boxlen),
                        ),
                    );

                    canvas.draw(
                        &redcirc,
                        Vec2::new(
                            self.partilist[0][0] * (WINSIZE / self.boxlen),
                            (self.boxlen - self.partilist[0][1]) * (WINSIZE / self.boxlen),
                        ),
                    );
                }
            }
            // Draw text box of parameters and draw to screen
            let time = format!("Time: {}", self.time);
            canvas.draw(&graphics::Text::new(time), Vec2::new(WINSIZE + 10.0, 10.0));

            let speed_str = format!("Speed: {:.2}", self.speed);
            canvas.draw(
                &graphics::Text::new(speed_str),
                Vec2::new(WINSIZE + 10.0, 30.0),
            );

            let noise_str = format!("Noise: {:.1}", self.noise);
            canvas.draw(
                &graphics::Text::new(noise_str),
                Vec2::new(WINSIZE + 10.0, 50.0),
            );

            let partinum_str = format!("Num of Particles: {:.1}", self.partilist.len());
            canvas.draw(
                &graphics::Text::new(partinum_str),
                Vec2::new(WINSIZE + 10.0, 70.0),
            );

            let boxlen_str = format!("Box Length: {:.1}", self.boxlen);
            canvas.draw(
                &graphics::Text::new(boxlen_str),
                Vec2::new(WINSIZE + 10.0, 90.0),
            );
            let density_str = format!(
                "Density: {:.2}",
                (self.partilist.len() as f32 / (self.boxlen * self.boxlen))
            );
            canvas.draw(
                &graphics::Text::new(density_str),
                Vec2::new(WINSIZE + 10.0, 110.0),
            );

            // let target_fps = format!("Target FPS: {:.1}", self.fps);
            // canvas.draw(
            //     &graphics::Text::new(target_fps),
            //     Vec2::new(WINSIZE + 10.0, 130.0),
            // );

            let actual_fps = format!("FPS: {:.1}", ggez::timer::fps(ctx));
            canvas.draw(
                &graphics::Text::new(actual_fps),
                Vec2::new(WINSIZE + 10.0, 150.0),
            );
            let normvel_str = format!("Norm vel: {:.3}", self.normvel);
            canvas.draw(
                &graphics::Text::new(normvel_str),
                Vec2::new(WINSIZE + 10.0, 190.0),
            );
            // Drawing surrounding box
            let rect = graphics::Rect::new(WINSIZE, 0.0, 3.0, WINSIZE);
            canvas.draw(
                &graphics::Quad,
                graphics::DrawParam::new()
                    .dest(rect.point())
                    .scale(rect.size())
                    .color(Color::WHITE),
            );
            let rect = graphics::Rect::new(0.0, 0.0, WINSIZE, 3.0);
            canvas.draw(
                &graphics::Quad,
                graphics::DrawParam::new()
                    .dest(rect.point())
                    .scale(rect.size())
                    .color(Color::WHITE),
            );
            let rect = graphics::Rect::new(0.0, 0.0, 3.0, WINSIZE);
            canvas.draw(
                &graphics::Quad,
                graphics::DrawParam::new()
                    .dest(rect.point())
                    .scale(rect.size())
                    .color(Color::WHITE),
            );
            let rect = graphics::Rect::new(0.0, WINSIZE - 3.0, WINSIZE, 3.0);
            canvas.draw(
                &graphics::Quad,
                graphics::DrawParam::new()
                    .dest(rect.point())
                    .scale(rect.size())
                    .color(Color::WHITE),
            );
            let elapsed = now.elapsed();
            if self.time % 100 == 0 {
                println!("Drawing: {:.2?}", elapsed);
            }

            // Finish Drawing
            canvas.finish(ctx)?;
        }
        Ok(())
    }
}

pub fn main() -> GameResult {
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();

    use ggez::conf;
    let cb = ggez::ContextBuilder::new("Vicsek", "Rob Smith")
        .window_setup(conf::WindowSetup::default().title("Vicsek Demo"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(WINSIZE + 300.0, WINSIZE));
    let (mut ctx, event_loop) = cb.build()?;
    let state = MainState::new(&mut ctx)?;
    event::run(ctx, event_loop, state)
}
