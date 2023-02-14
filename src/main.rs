use ggez::{
    graphics::Rect,
    input::{
        self,
        keyboard::{self, KeyInput},
    },
};
// To run at build speed
// test
// cargo run --release -- -C target=cpu_native
#[allow(deprecated)]
//// Imports
use ggez::{
    event,
    glam::*,
    graphics::{self, Color, DrawMode, DrawParam, Mesh, MeshBuilder, PxScale},
    input::keyboard::KeyCode,
    mint::Point2,
    timer, Context, GameResult,
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    time::Instant,
};

//// Functions
fn randpartigen(boxlen: f32) -> Vec<f32> {
    // Generate a random position and bearning inside the box
    let mut rng: ThreadRng = rand::thread_rng();
    let coords: Vec<f32> = vec![
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(2. * PI)) as f32,
    ];
    return coords;
}

fn box_nn(boxcoord: Vec<i32>, boxlen: f32) -> Vec<Vec<i32>> {
    // Gives surrounding 3x3 grid given the current box including periodic boundry conditions
    // box is labeled as [x,y]
    let boxlen: i32 = (boxlen) as i32;
    let x: i32 = boxcoord[0];
    let y: i32 = boxcoord[1];
    let nnlist: Vec<Vec<i32>> = vec![
        vec![
            (x - 1 as i32).rem_euclid(boxlen),
            (y - 1 as i32).rem_euclid(boxlen),
        ],
        vec![
            (x as i32).rem_euclid(boxlen),
            (y - 1 as i32).rem_euclid(boxlen),
        ],
        vec![
            (x + 1 as i32).rem_euclid(boxlen),
            (y - 1 as i32).rem_euclid(boxlen),
        ],
        vec![
            (x - 1 as i32).rem_euclid(boxlen),
            (y as i32).rem_euclid(boxlen),
        ],
        vec![(x as i32).rem_euclid(boxlen), (y as i32).rem_euclid(boxlen)],
        vec![
            (x + 1 as i32).rem_euclid(boxlen),
            (y as i32).rem_euclid(boxlen),
        ],
        vec![
            (x - 1 as i32).rem_euclid(boxlen),
            (y + 1 as i32).rem_euclid(boxlen),
        ],
        vec![
            (x as i32).rem_euclid(boxlen),
            (y + 1 as i32).rem_euclid(boxlen),
        ],
        vec![
            (x + 1 as i32).rem_euclid(boxlen),
            (y + 1 as i32).rem_euclid(boxlen),
        ],
    ];
    return nnlist;
}

fn new_angle_for_index_array(
    // The angle update to be parallalised if vicsek flag == 1 perform standard vicsek if nnflag == 1 consider nearest neighbours
    partialhashmap: HashMap<Vec<i32>, Vec<usize>>,
    particlelist: &Vec<Vec<f32>>,
    boxlen: f32,
    noise: f32,
    vicsekflag: bool,
    blindangle: f32,
    maxrotangle: f32,
    nninteractionnum: u32,
    nnflag: bool,
) -> Vec<Vec<f32>> {
    let mut new_angle_for_index_array: Vec<Vec<f32>> = Vec::new();
    // Go though each boxcoord given and append angles and indicies to the output array
    for (currentboxcoord, currentboxcoordpartiindicies) in partialhashmap.iter() {
        // Get parti indicies from surrounding boxes
        let mut indecies_to_check: Vec<usize> = Vec::new();
        for checkingbox in &mut box_nn(currentboxcoord.to_vec(), boxlen).into_iter() {
            if partialhashmap.contains_key(&checkingbox) {
                indecies_to_check.extend(partialhashmap.get(&checkingbox).unwrap());
            }
        }
        // Update angle for every particle for current box
        for activepartiindex in currentboxcoordpartiindicies {
            let mut sum_of_sin: f32 = 0.0;
            let mut sum_of_cos: f32 = 0.0;
            let mut rng: ThreadRng = rand::thread_rng();

            let currentparti: &Vec<f32> = &particlelist[*activepartiindex];
            let currentpartix: f32 = currentparti[0];
            let currentpartiy: f32 = currentparti[1];
            let currentpartia: f32 = currentparti[2];

            // If nn is on this will collect all particles within conditions
            let mut parti_nnlist: Vec<Vec<f32>> = Vec::new();
            for otherindex in &indecies_to_check {
                // vicsek conditions
                if vicsekflag == true {
                    let otherparti: &Vec<f32> = &particlelist[*otherindex];
                    let otherpartix: f32 = otherparti[0];
                    let otherpartiy: f32 = otherparti[1];
                    let otherpartia: f32 = otherparti[2];

                    let mut dx: f32 = (currentpartix - otherpartix).abs();
                    let mut dy: f32 = (currentpartiy - otherpartiy).abs();

                    // Adjusting distance for periodic boundry conditions
                    if dx > boxlen / 2.0 {
                        dx = boxlen - dx;
                    }
                    if dy > boxlen / 2.0 {
                        dy = boxlen - dy;
                    }

                    let distsquared: f32 = dx * dx + dy * dy;

                    if distsquared < 1.0 {
                        if nnflag == true {
                            parti_nnlist.push(vec![distsquared, otherpartia]);
                        } else {
                            sum_of_sin += otherpartia.sin();
                            sum_of_cos += otherpartia.cos();
                        }
                    }

                    // milling conditions
                } else {
                    let otherparti: &Vec<f32> = &particlelist[*otherindex];
                    let otherpartix: f32 = otherparti[0];
                    let otherpartiy: f32 = otherparti[1];
                    let otherpartia: f32 = otherparti[2];

                    let velx: f32 = currentpartia.sin();
                    let vely: f32 = currentpartia.cos();
                    let mut distx: f32 = otherpartix - currentpartix;
                    let mut disty: f32 = otherpartiy - currentpartiy;

                    // 3x3 box coord correction with periodic boundry conditions
                    if (distx).abs() > 2.0 {
                        if otherpartix > currentpartix {
                            distx -= boxlen;
                        } else {
                            distx += boxlen;
                        }
                    }
                    if (disty).abs() > 2.0 {
                        if otherpartiy > currentpartiy {
                            disty -= boxlen;
                        } else {
                            disty += boxlen;
                        }
                    }
                    let distsquared: f32 = (distx * distx) + (disty * disty);
                    // This condition is required so that the arctan is well defined
                    if distsquared == 0. {
                        if nnflag == true {
                            parti_nnlist.push(vec![distsquared, otherpartia]);
                        } else {
                            sum_of_sin += otherpartia.sin();
                            sum_of_cos += otherpartia.cos();
                        }
                    }
                    // Condition to check if within radius
                    else if distsquared < 1.0 {
                        // Using dotproduct to calculate difference in angle
                        let dotprod: f32 = ((distx * velx) + (disty * vely)) / (distsquared).sqrt();
                        let diffanglerad: f32 = dotprod.acos();
                        // Conditions to check if within viewing angle
                        if diffanglerad < (180. - blindangle / 2.0) * (PI / 180.) {
                            if nnflag == true {
                                parti_nnlist.push(vec![distsquared, otherpartia]);
                            } else {
                                sum_of_sin += otherpartia.sin();
                                sum_of_cos += otherpartia.cos();
                            }
                        }
                    }
                }
            }

            // Use nn list to only add nearest n particles
            if nnflag == true {
                // Sorting nnlist so that neasest parti is closest
                parti_nnlist.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));
                // Adding the angles of the nearest n particles
                for i in 0..(nninteractionnum as usize + 1) {
                    // Fixing crashes
                    if parti_nnlist.len() == 0 {
                        break;
                    }
                    // Only call the nearest neighbor if the index exists
                    if i > parti_nnlist.len() - 1 {
                        break;
                    };
                    sum_of_sin += parti_nnlist[i][1].sin();
                    sum_of_cos += parti_nnlist[i][1].cos();
                }
            }

            // Create a new angle from the interaction
            let mut arctan: f32 = f32::atan2(sum_of_sin, sum_of_cos);
            let maxangle: f32 = maxrotangle * (PI / 180.);

            // We modify the arctan so it cant be larger than the max angle
            if vicsekflag == false {
                let mut avangle: f32 = arctan.rem_euclid(2. * PI);
                let curangle: f32 = currentpartia.rem_euclid(2. * PI);

                // Logic to get the correct modular arithmatic of angles
                if avangle > curangle {
                    let case1: f32 = avangle - curangle;
                    let case2: f32 = (2. * PI) + curangle - avangle;
                    if case1 < case2 {
                        if case1 > maxangle {
                            avangle = curangle + maxangle;
                        }
                    }
                    if case2 < case1 {
                        if case2 > maxangle {
                            avangle = (curangle - maxangle).rem_euclid(2. * PI);
                        }
                    }
                }
                if curangle > avangle {
                    let case1: f32 = curangle - avangle;
                    let case2: f32 = (2. * PI) + avangle - curangle;
                    if case1 < case2 {
                        if case1 > maxangle {
                            avangle = curangle - maxangle;
                        }
                    }
                    if case2 < case1 {
                        if case2 > maxangle {
                            avangle = (curangle + maxangle).rem_euclid(2. * PI);
                        }
                    }
                }
                // Updating the arctan to the limited angle
                arctan = avangle;
            }

            // Include random angle
            let mut randangle: f32 = 0.0;
            if noise > 0.0 {
                randangle += rng.gen_range((-noise / 2.0)..(noise / 2.0)) as f32;
            }

            let updatedangle: f32 = arctan + randangle;

            // println!("");
            // println!("rand{:?}", randangle);
            // println!("arctan{:?}", arctan);
            // println!("updateangle{:?}", updatedangle);
            // println!("activeparti{:?}", activepartiindex);
            let new_angle_for_index: Vec<f32> = vec![*activepartiindex as f32, updatedangle];
            new_angle_for_index_array.push(new_angle_for_index)
        }
    }
    return new_angle_for_index_array;
}

fn convert_hsv_to_rgb(pixel: &Vec<f32>) -> [f32; 3] {
    // To convert angles into colors
    let [hue, saturation, value] = [pixel[0], pixel[1] / 100., pixel[2] / 100.];
    let max: f32 = value;
    let c: f32 = saturation * value;
    let min: f32 = max - c;
    let h_prime: f32 = if hue >= 300. {
        (hue - 360.) / 60.
    } else {
        hue / 60.
    };
    let (r, g, b) = match h_prime {
        x if -1. <= x && x < 1. => {
            if h_prime < 0. {
                (max, min, min - h_prime * c)
            } else {
                (max, min + h_prime * c, min)
            }
        }
        x if 1. <= x && x < 3. => {
            if h_prime < 2. {
                (min - (h_prime - 2.) * c, max, min)
            } else {
                (min, max, min + (h_prime - 2.) * c)
            }
        }
        x if 3. <= x && x < 5. => {
            if h_prime < 4. {
                (min, min - (h_prime - 4.) * c, max)
            } else {
                (min + (h_prime - 4.) * c, min, max)
            }
        }
        _ => unreachable!(),
    };
    [r, g, b]
}

fn semicirclearc(angle: f32, boxlen: f32) -> [Point2<f32>; 361] {
    let mut points = [Point2::from([0.0, 0.0]); 361];

    for i in 0..360 {
        points[i] = Point2::from([
            0.0 + (((360. - angle) / 360.) * (PI / 180.) * (i as f32) as f32).sin() * WINSIZE
                / (boxlen * 2.1),
            0.0 + (((360. - angle) / 360.) * (PI / 180.) * (i as f32) as f32).cos() * WINSIZE
                / (boxlen * 2.1),
        ])
    }
    return points;
}

// returns NN index list
fn RangeQuery(
    currentindex: usize,
    partilist: &Vec<Vec<f32>>,
    hashmap: &HashMap<Vec<i32>, Vec<usize>>,
    epsilon: f32,
    boxlen: f32,
) -> Vec<usize> {
    let mut neighbourslist = Vec::new();
    let boxindx = vec![
        partilist[currentindex as usize][0].floor() as i32,
        partilist[currentindex as usize][1].floor() as i32,
    ];

    // gather indecies from surrounding boxes
    let mut indecies_to_check: Vec<usize> = Vec::new();
    for checkingbox in &mut box_nn(boxindx, boxlen).into_iter() {
        if hashmap.contains_key(&checkingbox) {
            indecies_to_check.extend(hashmap.get(&checkingbox).unwrap());
        }
    }

    let currentpartix: f32 = partilist[currentindex][0];
    let currentpartiy: f32 = partilist[currentindex][1];

    for otherindex in indecies_to_check {
        let otherparti: &Vec<f32> = &partilist[otherindex];
        let otherpartix: f32 = otherparti[0];
        let otherpartiy: f32 = otherparti[1];
        let mut dx: f32 = (currentpartix - otherpartix).abs();
        let mut dy: f32 = (currentpartiy - otherpartiy).abs();

        if dx > boxlen / 2.0 {
            dx = boxlen - dx;
        }
        if dy > boxlen / 2.0 {
            dy = boxlen - dy;
        }

        let distsquared: f32 = dx * dx + dy * dy;
        if distsquared.sqrt() < epsilon {
            neighbourslist.push(otherindex);
        }
    }
    return neighbourslist;
}

fn DBSCAN(
    partilist: &Vec<Vec<f32>>,
    hashmap: &HashMap<Vec<i32>, Vec<usize>>,
    epsilon: f32,
    nmin: i32,
    boxlen: f32,
) -> HashMap<i32, Vec<usize>> {
    // ) -> Vec<_> {
    let mut clusterlist: Vec<i32> = std::iter::repeat(0)
        .take(partilist.len())
        .collect::<Vec<_>>();

    // Perform DBSCAN on the partilist using the hashmap to speed up range finding to populate clusterlist
    let mut cluster = 0;
    for currentindex in 0..partilist.len() {
        if clusterlist[currentindex] != 0 {
            continue;
        }
        let mut neighbours = RangeQuery(currentindex, &partilist, &hashmap, epsilon, boxlen);

        if neighbours.len() < nmin as usize {
            clusterlist[currentindex] = -1;
            continue;
        }
        cluster += 1;
        clusterlist[currentindex] = cluster;

        let mut Seed = neighbours;
        Seed.retain(|value| *value != currentindex);

        let mut cur = 0;
        while cur < Seed.len() {
            let Q = Seed[cur];
            if clusterlist[Q] == -1 {
                clusterlist[Q] = cluster;
            }
            if clusterlist[Q] != 0 {
                cur += 1;
                continue;
            }
            clusterlist[Q] = cluster;

            neighbours = RangeQuery(Q, &partilist, &hashmap, epsilon, boxlen);
            if neighbours.len() >= nmin as usize {
                for val in &neighbours {
                    Seed.push(*val);
                }
            }
            cur += 1;
        }
    }

    // Modify clusterlist to give a vector decending cluster size, might at somepoint be able to put this in DBSCAN code
    let mut clustermap: HashMap<i32, Vec<usize>> = HashMap::new();
    for index in 0..partilist.len() {
        let elem = clusterlist[index];
        clustermap.entry(elem).or_insert(Vec::new()).push(index);
    }

    // let mut vec: Vec<_> = tempmap.into_iter().collect();
    // vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    // let sorted_map: Vec<_> = vec.into_iter().collect();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // return vec![(1, vec![3 as usize])];
    return clustermap;
}

fn comcalc(indecieslist: &Vec<usize>, partilist: &Vec<Vec<f32>>, boxlen: f32) -> Vec<f32> {
    // x should be outer
    // y should be inner
    let mut outer_xsum = 0.;
    let mut outer_ysum = 0.;
    let mut inner_xsum = 0.;
    let mut inner_ysum = 0.;
    for index in indecieslist {
        // Projection onto torus
        let outerangle = partilist[*index][0] * (2. * PI) / boxlen;
        let innerangle = partilist[*index][1] * (2. * PI) / boxlen;
        outer_xsum += outerangle.sin();
        outer_ysum += outerangle.cos();
        inner_xsum += innerangle.sin();
        inner_ysum += innerangle.cos();
    }

    // Normalising and converting back to flat space
    let mut xcom = f32::atan2(
        (outer_ysum / ((outer_xsum * outer_xsum) + (outer_ysum * outer_ysum)).sqrt()),
        (outer_xsum / ((outer_xsum * outer_xsum) + (outer_ysum * outer_ysum)).sqrt()),
    ) * (180. / PI);
    // println!("length{:?}", (xcom));
    if xcom >= 0. && xcom <= 90. {
        xcom = 90. - xcom;
    }
    if xcom >= 90. && xcom <= 180. {
        xcom = 270. + (180. - xcom);
    }
    if xcom <= 0. && xcom >= -90. {
        xcom = 90. - xcom;
    }
    if xcom <= -90. && xcom >= -180. {
        xcom = 270. - (180. + xcom)
    }
    xcom = xcom * (PI / 180.) * (boxlen / (2. * PI));

    let mut ycom = f32::atan2(
        (inner_ysum / ((inner_xsum * inner_xsum) + (inner_ysum * inner_ysum)).sqrt()),
        (inner_xsum / ((inner_xsum * inner_xsum) + (inner_ysum * inner_ysum)).sqrt()),
    ) * (180. / PI);
    // println!("length{:?}", (xcom));
    if ycom >= 0. && ycom <= 90. {
        ycom = 90. - ycom;
    }
    if ycom >= 90. && ycom <= 180. {
        ycom = 270. + (180. - ycom);
    }
    if ycom <= 0. && ycom >= -90. {
        ycom = 90. - ycom;
    }
    if ycom <= -90. && ycom >= -180. {
        ycom = 270. - (180. + ycom)
    }
    ycom = ycom * (PI / 180.) * (boxlen / (2. * PI));

    return vec![xcom, ycom];
}
// Constants
const WINSIZE: f32 = 900.0;
// const TIMEDEBUG: i32 = 0;
const CPU_NUM: i32 = 8;
const PI: f32 = std::f32::consts::PI;

struct MainState {
    time: i32,
    noise: f32,
    boxlen: f32,
    blindangle: f32,
    speed: f32,
    partilist: Vec<Vec<f32>>,
    normvel: f32,
    fps: u32,
    visualiseflag: bool,
    stateupdateflag: u32,
    colorflag: bool,
    vicsekflag: bool,
    maxrotangle: f32,
    colorwheel: graphics::Image,
    nninteraction: u32,
    nnflag: bool,
    clusterflag: bool,
    dbscannmin: i32,
    dbscanepsilon: f32,
    freezeflag: bool,
    clustermap: HashMap<i32, Vec<usize>>,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<MainState> {
        let initpartinum: i32 = 2;
        let initboxlen: f32 = 20.0;

        // Generate list of random particles
        let mut parti_list = Vec::new();
        for _ in 0..initpartinum {
            parti_list.push(randpartigen(initboxlen));
        }

        let cwheel = graphics::Image::from_path(ctx, "/pngwing.com.png")?;

        Ok(MainState {
            time: 0,
            noise: 0.4,
            speed: 0.03,
            blindangle: 180.,
            partilist: parti_list,
            boxlen: initboxlen,
            normvel: 1.0,
            fps: 30,
            stateupdateflag: 0,
            visualiseflag: true,
            colorflag: false,
            vicsekflag: true,
            maxrotangle: 8.,
            colorwheel: cwheel,
            nninteraction: 1,
            nnflag: false,
            clusterflag: false,
            dbscannmin: 0,
            dbscanepsilon: 0.5,
            freezeflag: false,
            clustermap: HashMap::new(),
        })
    }
}

impl event::EventHandler<ggez::GameError> for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        // fix frame rate of update
        let desired_fps: u32 = self.fps;
        while timer::check_update_time(_ctx, desired_fps) {
            if self.freezeflag == false {
                self.time = self.time + 1;
                // fix frame rate of Drawing
                self.stateupdateflag = 1;

                // Check for any key press
                let k_ctx = &_ctx.keyboard;

                // Paramater controlls
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
                    if self.speed < 0.00 {
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

                // Insert new particles
                if k_ctx.is_key_pressed(KeyCode::Q) {
                    let mut rng = rand::thread_rng();
                    let newparti = vec![
                        rng.gen_range((0.0)..(self.boxlen)) as f32,
                        rng.gen_range((0.0)..(self.boxlen)) as f32,
                        rng.gen_range((0.0)..(2. * PI)) as f32,
                    ];
                    self.partilist.push(newparti);

                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        for _ in 0..9 {
                            let mut rng = rand::thread_rng();
                            let newparti = vec![
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(2. * PI)) as f32,
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
                                rng.gen_range((0.0)..(2. * PI)) as f32,
                            ];
                            self.partilist.push(newparti);
                        }
                    }
                }

                // Remove particles
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

                // change box length and invarient scale
                if k_ctx.is_key_pressed(KeyCode::E) {
                    self.boxlen += 1.0;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.boxlen -= 1.0;
                        let density = self.partilist.len() as f32 / (self.boxlen * self.boxlen);
                        self.boxlen += 1.0;
                        let numofparti = (density * self.boxlen * self.boxlen).ceil();

                        for _ in 0..self.partilist.len() {
                            self.partilist.pop();
                        }

                        for _ in 0..(numofparti as i32) {
                            let mut rng = rand::thread_rng();
                            let newparti = vec![
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(2. * PI)) as f32,
                            ];
                            self.partilist.push(newparti);
                        }
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::R) {
                    self.boxlen -= 1.0;
                    if self.boxlen < 4.0 {
                        self.boxlen = 4.0
                    }
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.boxlen += 1.0;
                        let density = self.partilist.len() as f32 / (self.boxlen * self.boxlen);
                        self.boxlen -= 1.0;
                        let numofparti = (density * self.boxlen * self.boxlen).ceil();

                        for _ in 0..self.partilist.len() {
                            self.partilist.pop();
                        }

                        for _ in 0..(numofparti as i32) {
                            let mut rng = rand::thread_rng();
                            let newparti = vec![
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(self.boxlen)) as f32,
                                rng.gen_range((0.0)..(2. * PI)) as f32,
                            ];
                            self.partilist.push(newparti);
                        }
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::A) {
                    self.blindangle += 1.;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.blindangle += 10.
                    }
                    if self.blindangle > 360.0 {
                        self.blindangle = 360.0
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::S) {
                    self.blindangle -= 1.;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.blindangle -= 10.
                    }
                    if self.blindangle < 0.0 {
                        self.blindangle = 0.0
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::D) {
                    self.maxrotangle += 1.;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.maxrotangle += 10.
                    }
                    if self.maxrotangle > 180.0 {
                        self.maxrotangle = 180.0
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::F) {
                    self.maxrotangle -= 1.;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.maxrotangle -= 10.
                    }
                    if self.maxrotangle < 0.0 {
                        self.maxrotangle = 0.0
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::Z) {
                    self.nninteraction += 1;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.nninteraction += 5;
                    }
                }
                if k_ctx.is_key_pressed(KeyCode::X) {
                    self.nninteraction -= 1;
                    if k_ctx.is_mod_active(ggez::input::keyboard::KeyMods::SHIFT) {
                        self.nninteraction -= 5;
                    }
                    if self.nninteraction < 1 {
                        self.nninteraction = 1
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::C) {
                    self.fps += 1;
                    if self.fps > 30 {
                        self.fps = 30
                    }
                    // }
                }
                if k_ctx.is_key_pressed(KeyCode::V) {
                    self.fps -= 1;
                    if self.fps < 1 {
                        self.fps = 1
                    }
                }

                //functions
                // Inset a partilce in the center
                if k_ctx.is_key_pressed(KeyCode::H) {
                    let mut rng = rand::thread_rng();
                    let newparti = vec![
                        self.boxlen / 2.0,
                        self.boxlen / 2.0,
                        rng.gen_range((0.0)..(2. * PI)) as f32,
                    ];
                    self.partilist.push(newparti);
                }

                // Randomise positions of all active particles
                if k_ctx.is_key_pressed(KeyCode::J) {
                    let len = self.partilist.len();
                    for _ in 0..len {
                        self.partilist.pop();
                    }
                    for _ in 0..len {
                        let mut rng = rand::thread_rng();
                        let newparti = vec![
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(self.boxlen)) as f32,
                            rng.gen_range((0.0)..(2. * PI)) as f32,
                        ];
                        self.partilist.push(newparti);
                    }
                }

                // Remove all particles
                if k_ctx.is_key_pressed(KeyCode::T) {
                    for _ in 0..self.partilist.len() {
                        self.partilist.pop();
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::Key1) {
                    self.dbscannmin += 1;
                }

                if k_ctx.is_key_pressed(KeyCode::Key2) {
                    self.dbscannmin -= 1;
                    if self.dbscannmin < 0 {
                        self.dbscannmin = 0
                    }
                }

                if k_ctx.is_key_pressed(KeyCode::Key3) {
                    self.dbscanepsilon += 0.01;
                }
                if k_ctx.is_key_pressed(KeyCode::Key4) {
                    self.dbscanepsilon -= 0.01;
                    if self.dbscanepsilon < 0.01 {
                        self.dbscanepsilon = 0.01
                    }
                }

                ///////////////////////////////////////////////////////////////////////
                // Calculate Normalised Velocity every 10 steps
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

                // Update the position of the particles with periodic boundry conditions
                for partiindex in 0..(self.partilist.len()) {
                    self.partilist[partiindex][0] +=
                        self.partilist[partiindex][2].sin() * self.speed;
                    self.partilist[partiindex][1] +=
                        self.partilist[partiindex][2].cos() * self.speed;
                    if self.partilist[partiindex][0] < 0.0 {
                        self.partilist[partiindex][0] += self.boxlen
                    };
                    if self.partilist[partiindex][0] > self.boxlen {
                        self.partilist[partiindex][0] -= self.boxlen
                    };
                    if self.partilist[partiindex][1] < 0.0 {
                        self.partilist[partiindex][1] += self.boxlen
                    };
                    if self.partilist[partiindex][1] > self.boxlen {
                        self.partilist[partiindex][1] -= self.boxlen
                    };
                }

                // Creating Hash map
                let mut boxhashmap = HashMap::new();
                for parti_indx in 0..self.partilist.len() {
                    // Assigning each particle to a box
                    let boxindx = vec![
                        self.partilist[parti_indx as usize][0].floor() as i32,
                        self.partilist[parti_indx as usize][1].floor() as i32,
                    ];

                    boxhashmap
                        .entry(boxindx)
                        .or_insert(Vec::new())
                        .push(parti_indx);
                }

                // Calculate the clustering
                if self.clusterflag == true {
                    if self.dbscannmin > 0 {
                        self.clustermap = DBSCAN(
                            &self.partilist,
                            &boxhashmap,
                            self.dbscanepsilon,
                            self.dbscannmin,
                            self.boxlen,
                        );
                    } else {
                        // when nmin is 0 we consider everything as a single cluster
                        let mut everyindexhasmap = HashMap::new();
                        for parti_indx in 0..self.partilist.len() {
                            everyindexhasmap
                                .entry(1)
                                .or_insert(Vec::new())
                                .push(parti_indx);
                        }
                        self.clustermap = everyindexhasmap;
                    }
                }

                // Spltting Hashmap into small sections to be parallalized
                let hashmaplist: Vec<_> = boxhashmap.into_iter().collect();
                let chunk_len = (hashmaplist.len() / CPU_NUM as usize) as usize + 1;
                let chunks: Vec<HashMap<_, _>> = hashmaplist
                    .chunks(chunk_len)
                    .map(|c| c.iter().cloned().collect())
                    .collect();

                // Updating angles in parallal
                let parallellist: Vec<Vec<Vec<f32>>> = chunks
                    .par_iter()
                    .map(|chunk| {
                        new_angle_for_index_array(
                            chunk.clone(),
                            &self.partilist,
                            self.boxlen,
                            self.noise,
                            self.vicsekflag,
                            self.blindangle,
                            self.maxrotangle,
                            self.nninteraction,
                            self.nnflag,
                        )
                    })
                    .collect();
                let flattenlist = parallellist.into_iter().flatten().collect::<Vec<_>>();

                // Updating partilist with new angles
                for change in flattenlist {
                    self.partilist[change[0] as usize][2] = change[1]
                }
                // println!("{:?}", self.partilist)
            }
        }

        Ok(())
    }

    // Flag buttons
    fn key_down_event(&mut self, _ctx: &mut Context, input: KeyInput, repeat: bool) -> GameResult {
        if !repeat {
            if input.keycode == Some(keyboard::KeyCode::T) {
                self.visualiseflag = !self.visualiseflag;
            };
            if input.keycode == Some(keyboard::KeyCode::Y) {
                self.colorflag = !self.colorflag;
            };
            if input.keycode == Some(keyboard::KeyCode::U) {
                self.vicsekflag = !self.vicsekflag;
            };
            if input.keycode == Some(keyboard::KeyCode::I) {
                self.nnflag = !self.nnflag;
            };
            if input.keycode == Some(keyboard::KeyCode::O) {
                self.clusterflag = !self.clusterflag;
            };
            if input.keycode == Some(keyboard::KeyCode::P) {
                self.freezeflag = !self.freezeflag;
            };
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        // To get draw and update at same fps
        if self.stateupdateflag == 1 {
            self.stateupdateflag = 0;

            // Set color of the canvas
            let mut canvas =
                graphics::Canvas::from_frame(ctx, graphics::Color::from([0.1, 0.2, 0.3, 1.0]));

            // Check if we have to draw particles to screen
            if self.visualiseflag == true {
                if self.partilist.len() > 0 {
                    // Black background for squares
                    let highlighttest = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::fill(),
                        ggez::graphics::Rect::new(0., 0., WINSIZE, WINSIZE),
                        Color::BLACK,
                    )?;
                    canvas.draw(&highlighttest, DrawParam::default());

                    let mut mesh_builder = MeshBuilder::new();
                    for particle in self.partilist.iter() {
                        // Defualt color should be white
                        let mut particolor = Color::WHITE;
                        if self.colorflag == true {
                            let [r, g, b] = convert_hsv_to_rgb(&vec![
                                particle[2].rem_euclid(2. * PI) * (180. / PI),
                                100.0,
                                100.0,
                            ]);
                            particolor = Color::new(r, g, b, 1.0);
                        }
                        mesh_builder.circle(
                            DrawMode::fill(),
                            Vec2::new(
                                particle[0] * (WINSIZE / self.boxlen),
                                (self.boxlen - particle[1]) * (WINSIZE / self.boxlen),
                            ),
                            2.0,
                            10.0,
                            particolor, // Color::new(r, g, b, 1.0),
                        )?;

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
                            particolor,
                        )?;
                    }

                    // Draw collection of circles and lines to canvas
                    let mesh = Mesh::from_data(ctx, mesh_builder.build());
                    canvas.draw(&mesh, DrawParam::default());

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    // Color cluster particles and display COM as green circ

                    if self.clusterflag == true {
                        let tempmap = &self.clustermap;
                        let mut vec: Vec<_> = tempmap.into_iter().collect();
                        vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                        let sorted_map_clustermap: Vec<_> = vec.into_iter().collect();

                        let mut mesh_builder = MeshBuilder::new();
                        let mut cangle = 0.;
                        for sizeindex in 0..sorted_map_clustermap.len() {
                            let (key, values) = &sorted_map_clustermap[sizeindex];
                            let mut particolor = Color::WHITE;
                            if key != &&-1 {
                                let [r, g, b] = convert_hsv_to_rgb(&vec![
                                    ((cangle * 10.) as f32).rem_euclid(2. * PI) * (180. / PI),
                                    100.0,
                                    100.0,
                                ]);
                                particolor = Color::new(r, g, b, 1.0);
                                cangle += 10.;
                            }

                            for partiindex in *values {
                                mesh_builder.circle(
                                    DrawMode::fill(),
                                    Vec2::new(
                                        self.partilist[*partiindex][0] * (WINSIZE / self.boxlen),
                                        (self.boxlen - self.partilist[*partiindex][1])
                                            * (WINSIZE / self.boxlen),
                                    ),
                                    2.0,
                                    10.0,
                                    particolor, // Color::new(r, g, b, 1.0),
                                )?;
                                mesh_builder.line(
                                    &[
                                        Vec2::new(
                                            self.partilist[*partiindex][0]
                                                * (WINSIZE / self.boxlen),
                                            (self.boxlen - self.partilist[*partiindex][1])
                                                * (WINSIZE / self.boxlen),
                                        ),
                                        Vec2::new(
                                            (self.partilist[*partiindex][0]
                                                + self.partilist[*partiindex][2].sin())
                                                * (WINSIZE / self.boxlen),
                                            (self.boxlen
                                                - (self.partilist[*partiindex][1]
                                                    + self.partilist[*partiindex][2].cos()))
                                                * (WINSIZE / self.boxlen),
                                        ),
                                    ],
                                    1.0,
                                    particolor,
                                )?;
                            }

                            // COM Calculations

                            fn normvel_cluster(
                                values: &Vec<usize>,
                                partilist: &Vec<Vec<f32>>,
                            ) -> f32 {
                                let mut sumsin: f32 = 0.0;
                                let mut sumcos: f32 = 0.0;
                                for value in values {
                                    let parti = &partilist[*value];
                                    sumsin += &parti[2].sin();
                                    sumcos += &parti[2].cos();
                                }
                                let normvel = (1.0 / partilist.len() as f32)
                                    * f32::sqrt((sumsin * sumsin) + (sumcos * sumcos));
                                return normvel;
                            }

                            if key != &&-1 {
                                let com = comcalc(values, &self.partilist, self.boxlen);
                                let xcom = com[0];
                                let ycom = com[1];
                                let mut sum = 0.;
                                for value in *values {
                                    let parti = &self.partilist[*value];
                                    // let otherparti: &Vec<f32> = &particlelist[*otherindex];
                                    // TODO refactor this as a torus distance function
                                    let partix: f32 = parti[0];
                                    let partiy: f32 = parti[1];
                                    let partia: f32 = parti[2];

                                    let velx: f32 = partia.sin();
                                    let vely: f32 = partia.cos();
                                    let mut distx: f32 = partix - xcom;
                                    let mut disty: f32 = partiy - ycom;

                                    // 3x3 box coord correction with periodic boundry conditions
                                    if (distx).abs() > self.boxlen / 2. {
                                        if partix > xcom {
                                            distx -= self.boxlen;
                                        } else {
                                            distx += self.boxlen;
                                        }
                                    }
                                    if (disty).abs() > self.boxlen / 2. {
                                        if partiy > ycom {
                                            disty -= self.boxlen;
                                        } else {
                                            disty += self.boxlen;
                                        }
                                    }
                                    let distsquared: f32 = (distx * distx) + (disty * disty);

                                    let angmom = (((distx * vely) - (disty * velx)).abs())
                                        / distsquared.sqrt();
                                    sum += angmom;
                                }
                                let normangvel = sum / values.len() as f32;
                                let normvel = normvel_cluster(*values, &self.partilist);
                                // println!("{:?}", sum / values.len() as f32);

                                mesh_builder.circle(
                                    graphics::DrawMode::fill(),
                                    Vec2::new(
                                        xcom * (WINSIZE / self.boxlen),
                                        (self.boxlen - ycom) * (WINSIZE / self.boxlen),
                                    ),
                                    WINSIZE / (self.boxlen * 2.0),
                                    1.0,
                                    Color::GREEN,
                                )?;
                            };
                        }
                        let mesh = Mesh::from_data(ctx, mesh_builder.build());
                        canvas.draw(&mesh, DrawParam::default());
                    };

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    // // // mean velocity field test
                    // let mut boxhashmap = HashMap::new();
                    // for parti_indx in 0..self.partilist.len() {
                    //     // Assigning each particle to a box
                    //     let boxindx = vec![
                    //         self.partilist[parti_indx as usize][0].floor() as i32,
                    //         self.partilist[parti_indx as usize][1].floor() as i32,
                    //     ];

                    //     boxhashmap
                    //         .entry(boxindx)
                    //         .or_insert(Vec::new())
                    //         .push(self.partilist[parti_indx][2]);
                    // }

                    // let mut mesh_builder = MeshBuilder::new();
                    // for (currentboxcoord, _currentboxcoordpartiindicies) in boxhashmap.iter() {
                    //     //search surrounding boxes for gradients calculation
                    //     let boxestocheck = vec![
                    //         vec![
                    //             (currentboxcoord[0] + 1).rem_euclid(self.boxlen as i32),
                    //             (currentboxcoord[1] + 0).rem_euclid(self.boxlen as i32),
                    //         ],
                    //         vec![
                    //             (currentboxcoord[0] - 1).rem_euclid(self.boxlen as i32),
                    //             (currentboxcoord[1] + 0).rem_euclid(self.boxlen as i32),
                    //         ],
                    //         vec![
                    //             (currentboxcoord[0] + 0).rem_euclid(self.boxlen as i32),
                    //             (currentboxcoord[1] + 1).rem_euclid(self.boxlen as i32),
                    //         ],
                    //         vec![
                    //             (currentboxcoord[0] + 0).rem_euclid(self.boxlen as i32),
                    //             (currentboxcoord[1] - 1).rem_euclid(self.boxlen as i32),
                    //         ],
                    //     ];
                    //     let mut boxnext_xdir_yvel: f32 = 0.;
                    //     let mut boxnext_xdir_xvel: f32 = 0.;

                    //     let mut boxprev_xdir_yvel: f32 = 0.;
                    //     let mut boxprev_xdir_xvel: f32 = 0.;

                    //     let mut boxnext_ydir_xvel: f32 = 0.;
                    //     let mut boxnext_ydir_yvel: f32 = 0.;

                    //     let mut boxprev_ydir_xvel: f32 = 0.;
                    //     let mut boxprev_ydir_yvel: f32 = 0.;
                    //     // consider next x box
                    //     if boxhashmap.contains_key(&boxestocheck[0]) {
                    //         let anglesumlist = boxhashmap.get(&boxestocheck[0]).unwrap();
                    //         let anglesum: f32 = anglesumlist.iter().sum();

                    //         boxnext_xdir_yvel = (anglesum / anglesumlist.len() as f32).cos();
                    //         boxnext_xdir_xvel = (anglesum / anglesumlist.len() as f32).sin();
                    //     }
                    //     // consider prev x box
                    //     if boxhashmap.contains_key(&boxestocheck[1]) {
                    //         let anglesumlist = boxhashmap.get(&boxestocheck[1]).unwrap();
                    //         let anglesum: f32 = anglesumlist.iter().sum();

                    //         boxprev_xdir_yvel = (anglesum / anglesumlist.len() as f32).cos();
                    //         boxprev_xdir_xvel = (anglesum / anglesumlist.len() as f32).sin();
                    //     }
                    //     // consider next y box
                    //     if boxhashmap.contains_key(&boxestocheck[2]) {
                    //         let anglesumlist = boxhashmap.get(&boxestocheck[2]).unwrap();
                    //         let anglesum: f32 = anglesumlist.iter().sum();

                    //         boxnext_ydir_xvel = (anglesum / anglesumlist.len() as f32).sin();
                    //         boxnext_ydir_yvel = (anglesum / anglesumlist.len() as f32).cos();
                    //     }
                    //     // consider prev y box
                    //     if boxhashmap.contains_key(&boxestocheck[3]) {
                    //         let anglesumlist = boxhashmap.get(&boxestocheck[3]).unwrap();
                    //         let anglesum: f32 = anglesumlist.iter().sum();

                    //         boxprev_ydir_xvel = (anglesum / anglesumlist.len() as f32).sin();
                    //         boxprev_ydir_yvel = (anglesum / anglesumlist.len() as f32).cos();
                    //     }
                    //     let dvy_dx = boxnext_xdir_yvel - boxprev_xdir_yvel;
                    //     let dvx_dx = boxnext_xdir_xvel - boxprev_xdir_xvel;
                    //     let dvx_dy = boxnext_ydir_xvel - boxprev_ydir_xvel;
                    //     let dvy_dy = boxnext_ydir_yvel - boxprev_ydir_yvel;

                    //     let antisym = 0.5 * f32::powi((dvx_dy - dvy_dx), 2);
                    //     let sym = f32::powi(dvx_dx, 2)
                    //         + f32::powi(dvy_dy, 2)
                    //         + 0.5 * f32::powi(dvy_dx + dvx_dy, 2);

                    //     let mut Q = 0.5 * (antisym - sym);
                    //     let mut omega = antisym / (antisym + sym);

                    //     if Q > 1. {
                    //         println! {"{:?}", Q}
                    //         let [r, g, b] = convert_hsv_to_rgb(&vec![
                    //             Q.rem_euclid(2. * 3.14) * (180. / 3.14),
                    //             100.0,
                    //             100.0,
                    //         ]);

                    //         let particolor = Color::new(r, g, b, 0.6);
                    //         mesh_builder.rectangle(
                    //             graphics::DrawMode::fill(),
                    //             ggez::graphics::Rect::new(
                    //                 (currentboxcoord[0] as f32 * (WINSIZE / self.boxlen)),
                    //                 ((self.boxlen - currentboxcoord[1] as f32 - 1.)
                    //                     * (WINSIZE / self.boxlen)),
                    //                 (WINSIZE / self.boxlen),
                    //                 (WINSIZE / self.boxlen),
                    //             ),
                    //             particolor,
                    //         )?;
                    //     }
                    // }

                    // let mesh = Mesh::from_data(ctx, mesh_builder.build());
                    // canvas.draw(&mesh, DrawParam::default());

                    //// old stuff
                    // Highlight first point
                    let highlightfirst = graphics::Mesh::new_circle(
                        ctx,
                        graphics::DrawMode::fill(),
                        vec2(0., 0.),
                        WINSIZE / (self.boxlen * 2.0),
                        1.0,
                        Color::RED,
                    )?;
                    canvas.draw(
                        &highlightfirst,
                        Vec2::new(
                            self.partilist[0][0] * (WINSIZE / self.boxlen),
                            (self.boxlen - self.partilist[0][1]) * (WINSIZE / self.boxlen),
                        ),
                    );

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

                    // Green viewing angle
                    if self.vicsekflag == false {
                        let blindarc = graphics::Mesh::new_polygon(
                            ctx,
                            graphics::DrawMode::fill(),
                            &semicirclearc(self.blindangle, self.boxlen),
                            Color::YELLOW,
                        )?;
                        canvas.draw(
                            &blindarc,
                            DrawParam::new()
                                .dest(vec2(
                                    self.partilist[0][0] * (WINSIZE / self.boxlen),
                                    (self.boxlen - self.partilist[0][1]) * (WINSIZE / self.boxlen),
                                ))
                                .rotation(self.partilist[0][2] - self.blindangle * (PI / 360.)),
                        );
                    }
                }
            }

            // Draw text box of parameters and draw to screen
            let offsetfromedge = 50.;
            let secondcolumn = 380.;
            // add in second column
            canvas.draw(
                graphics::Text::new(format!("Parameter")).set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 10.0),
            );
            canvas.draw(
                graphics::Text::new(format!("Measurments")).set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 10.0),
            );
            canvas.draw(
                graphics::Text::new(format!("FPS: {:.1}", ggez::timer::fps(ctx)))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 40.0),
            );
            canvas.draw(
                graphics::Text::new(format!("Updates/s: {:.1}", self.fps))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 40.0),
            );

            canvas.draw(
                graphics::Text::new(format!("Time: {}", self.time)).set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 100.0),
            );

            canvas.draw(
                graphics::Text::new(format!(
                    "Density: {:.2}",
                    (self.partilist.len() as f32 / (self.boxlen * self.boxlen))
                ))
                .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 100.0),
            );

            canvas.draw(
                graphics::Text::new(format!("Speed: {:.2}", self.speed))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 130.0),
            );
            canvas.draw(
                graphics::Text::new(format!("Norm vel: {:.3}", self.normvel))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 130.0),
            );

            canvas.draw(
                graphics::Text::new(format!("Noise: {:.1}", self.noise))
                    .set_scale(PxScale::from(30.0)),
                Vec2::new(WINSIZE + offsetfromedge, 160.0),
            );
            canvas.draw(
                graphics::Text::new(format!("Box Length: {:.1}", self.boxlen))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 190.0),
            );
            canvas.draw(
                graphics::Text::new(format!("Num of Particles: {:.1}", self.partilist.len()))
                    .set_scale(PxScale::from(30.)),
                Vec2::new(WINSIZE + offsetfromedge, 220.0),
            );

            // Drawing color wheel
            if self.colorflag == true {
                canvas.draw(
                    &self.colorwheel,
                    graphics::DrawParam::new()
                        .dest([WINSIZE + offsetfromedge, 330.])
                        .scale([0.3, 0.3]),
                );
            }

            if self.vicsekflag == false {
                canvas.draw(
                    graphics::Text::new(format!("Blind angle: {:.1}", self.blindangle))
                        .set_scale(PxScale::from(30.0)),
                    Vec2::new(WINSIZE + offsetfromedge, 520.0),
                );

                canvas.draw(
                    graphics::Text::new(format!("Max rotation: {:.1}", self.maxrotangle))
                        .set_scale(PxScale::from(30.0)),
                    Vec2::new(WINSIZE + offsetfromedge, 550.0),
                );
            }

            if self.nnflag == true {
                canvas.draw(
                    graphics::Text::new(format!("Num of neighbours: {:.1}", self.nninteraction))
                        .set_scale(PxScale::from(30.)),
                    Vec2::new(WINSIZE + offsetfromedge, 610.0),
                );
            }

            if self.clusterflag == true {
                if self.partilist.len() > 0 {
                    canvas.draw(
                        graphics::Text::new(format!("Cluster nMin: {:.1}", self.dbscannmin))
                            .set_scale(PxScale::from(30.)),
                        Vec2::new(WINSIZE + offsetfromedge, 670.0),
                    );

                    // getting ordered cluster map for use in statistics

                    let tempmap = &self.clustermap;
                    let mut vec: Vec<_> = tempmap.into_iter().collect();
                    vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                    let sorted_map_clustermap: Vec<_> = vec.into_iter().collect();

                    let mut max_clus_size = self.partilist.len();
                    if sorted_map_clustermap.len() > 1 {
                        if sorted_map_clustermap[0].0 == &-1 {
                            max_clus_size = sorted_map_clustermap[1].1.len()
                        } else {
                            max_clus_size = sorted_map_clustermap[0].1.len()
                        }
                    }

                    let mut numclus = sorted_map_clustermap.len();
                    if self.clustermap.contains_key(&-1) {
                        numclus = sorted_map_clustermap.len() - 1;
                    }

                    // let mut numclusters =
                    canvas.draw(
                        graphics::Text::new(format!("Num of Clusters {:.1}", numclus))
                            .set_scale(PxScale::from(30.)),
                        Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 670.0),
                    );
                    canvas.draw(
                        graphics::Text::new(format!("Cluster min dist: {:.2}", self.dbscanepsilon))
                            .set_scale(PxScale::from(30.)),
                        Vec2::new(WINSIZE + offsetfromedge, 700.0),
                    );
                    // could create histogram of this would be quite cool
                    canvas.draw(
                        graphics::Text::new(format!(
                            "Max clus : {:.2}%",
                            100 * max_clus_size / self.partilist.len()
                        ))
                        .set_scale(PxScale::from(30.)),
                        Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 700.0),
                    );
                    let mut numnoise: usize = 0;
                    if self.clustermap.contains_key(&-1) {
                        numnoise = 100 * &self.clustermap.get(&-1).expect("msg").len()
                            / self.partilist.len();
                    }
                    canvas.draw(
                        graphics::Text::new(format!("Num Noise : {:.2}%", numnoise))
                            .set_scale(PxScale::from(30.)),
                        Vec2::new(WINSIZE + offsetfromedge + secondcolumn, 730.0),
                    );
                }
            }
            // let mut fullindicies = Vec::new();
            // for i in 0..self.partilist.len() {
            //     fullindicies.push(i);
            // }

            // Average circle thing
            // let xcenterouter = WINSIZE + 100.;
            // let ycenterouter = 820.;
            // let xcenterinner = WINSIZE + 300.;
            // let ycenterinner = 820.;
            // let rcirc = 50.;

            // let mut mesh_builder = MeshBuilder::new();
            // // draw outer circles x should be first
            // mesh_builder.circle(
            //     graphics::DrawMode::stroke(1.),
            //     Vec2::new(xcenterouter, ycenterouter),
            //     rcirc,
            //     1.0,
            //     Color::WHITE,
            // )?;
            // // mesh_builder.circle(
            // //     graphics::DrawMode::stroke(1.),
            // //     Vec2::new(xcenterinner, ycenterinner),
            // //     rcirc,
            // //     1.0,
            // //     Color::WHITE,
            // // )?;

            // let mut xsum = 0.;
            // let mut ysum = 0.;

            // for parti in &self.partilist {
            //     let tangle = parti[1] * (2. * PI) / self.boxlen;
            //     let xpos = xcenterouter + rcirc * tangle.sin();
            //     xsum += tangle.sin();
            //     let ypos = ycenterouter - rcirc * tangle.cos();
            //     ysum += tangle.cos();
            //     mesh_builder.circle(
            //         graphics::DrawMode::fill(),
            //         Vec2::new(xpos, ypos),
            //         4.,
            //         1.0,
            //         Color::WHITE,
            //     )?;
            // }

            // mesh_builder.circle(
            //     graphics::DrawMode::fill(),
            //     Vec2::new(
            //         xcenterouter + (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //         ycenterouter - (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //     ),
            //     6.,
            //     1.0,
            //     Color::GREEN,
            // )?;
            // mesh_builder.line(
            //     &[
            //         Vec2::new(xcenterouter, ycenterouter),
            //         Vec2::new(
            //             xcenterouter + (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //             ycenterouter - (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //         ),
            //     ],
            //     3.,
            //     Color::GREEN,
            // )?;

            // let tangle = self.partilist[0][1] * (2. * PI) / self.boxlen;
            // let xpos = xcenterouter + rcirc * tangle.sin();
            // let ypos = ycenterouter - rcirc * tangle.cos();
            // mesh_builder.circle(
            //     graphics::DrawMode::fill(),
            //     Vec2::new(xpos, ypos),
            //     4.,
            //     1.0,
            //     Color::RED,
            // )?;
            // let mesh = Mesh::from_data(ctx, mesh_builder.build());
            // canvas.draw(&mesh, DrawParam::default());

            // let mut ycom = f32::atan2(
            //     (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()),
            //     (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()),
            // ) * (180. / PI);
            // // println!("length{:?}", (xcom));
            // if ycom >= 0. && ycom <= 90. {
            //     ycom = 90. - ycom;
            // }
            // if ycom >= 90. && ycom <= 180. {
            //     ycom = 270. + (180. - ycom);
            // }
            // if ycom <= 0. && ycom >= -90. {
            //     ycom = 90. - ycom;
            // }
            // if ycom <= -90. && ycom >= -180. {
            //     ycom = 270. - (180. + ycom)
            // }
            // ycom = ycom * (PI / 180.) * (self.boxlen / (2. * PI));
            // let mut mesh_builder = MeshBuilder::new();
            // mesh_builder.circle(
            //     graphics::DrawMode::stroke(1.),
            //     Vec2::new(xcenterinner, ycenterinner),
            //     rcirc,
            //     1.0,
            //     Color::WHITE,
            // )?;

            // let mut xsum = 0.;
            // let mut ysum = 0.;

            // for parti in &self.partilist {
            //     let tangle = parti[0] * (2. * PI) / self.boxlen;
            //     let xpos = xcenterinner + rcirc * tangle.sin();
            //     xsum += tangle.sin();
            //     let ypos = ycenterinner - rcirc * tangle.cos();
            //     ysum += tangle.cos();
            //     mesh_builder.circle(
            //         graphics::DrawMode::fill(),
            //         Vec2::new(xpos, ypos),
            //         4.,
            //         1.0,
            //         Color::WHITE,
            //     )?;
            // }

            // mesh_builder.circle(
            //     graphics::DrawMode::fill(),
            //     Vec2::new(
            //         xcenterinner + (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //         ycenterinner - (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //     ),
            //     6.,
            //     1.0,
            //     Color::GREEN,
            // )?;
            // mesh_builder.line(
            //     &[
            //         Vec2::new(xcenterinner, ycenterinner),
            //         Vec2::new(
            //             xcenterinner + (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //             ycenterinner - (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()) * rcirc,
            //         ),
            //     ],
            //     3.,
            //     Color::GREEN,
            // )?;

            // let tangle = self.partilist[0][0] * (2. * PI) / self.boxlen;
            // let xpos = xcenterinner + rcirc * tangle.sin();
            // let ypos = ycenterinner - rcirc * tangle.cos();
            // mesh_builder.circle(
            //     graphics::DrawMode::fill(),
            //     Vec2::new(xpos, ypos),
            //     4.,
            //     1.0,
            //     Color::RED,
            // )?;
            // let mesh = Mesh::from_data(ctx, mesh_builder.build());
            // canvas.draw(&mesh, DrawParam::default());
            // let mut xcom = f32::atan2(
            //     (ysum / ((xsum * xsum) + (ysum * ysum)).sqrt()),
            //     (xsum / ((xsum * xsum) + (ysum * ysum)).sqrt()),
            // ) * (180. / PI);
            // // println!("length{:?}", (xcom));
            // if xcom >= 0. && xcom <= 90. {
            //     xcom = 90. - xcom;
            // }
            // if xcom >= 90. && xcom <= 180. {
            //     xcom = 270. + (180. - xcom);
            // }
            // if xcom <= 0. && xcom >= -90. {
            //     xcom = 90. - xcom;
            // }
            // if xcom <= -90. && xcom >= -180. {
            //     xcom = 270. - (180. + xcom)
            // }
            // xcom = xcom * (PI / 180.) * (self.boxlen / (2. * PI));

            // let com = graphics::Mesh::new_circle(
            //     ctx,
            //     graphics::DrawMode::fill(),
            //     vec2(0., 0.),
            //     WINSIZE / (self.boxlen * 2.0),
            //     1.0,
            //     Color::GREEN,
            // )?;
            // canvas.draw(
            //     &com,
            //     Vec2::new(
            //         xcom * (WINSIZE / self.boxlen),
            //         (self.boxlen - ycom) * (WINSIZE / self.boxlen),
            //     ),
            // );

            // // Drawing surrounding box
            // let rect = graphics::Rect::new(WINSIZE, 0.0, 3.0, WINSIZE);
            // canvas.draw(
            //     &graphics::Quad,
            //     graphics::DrawParam::new()
            //         .dest(rect.point())
            //         .scale(rect.size())
            //         .color(Color::WHITE),
            // );
            // let rect = graphics::Rect::new(0.0, 0.0, WINSIZE, 3.0);
            // canvas.draw(
            //     &graphics::Quad,
            //     graphics::DrawParam::new()
            //         .dest(rect.point())
            //         .scale(rect.size())
            //         .color(Color::WHITE),
            // );
            // let rect = graphics::Rect::new(0.0, 0.0, 3.0, WINSIZE);
            // canvas.draw(
            //     &graphics::Quad,
            //     graphics::DrawParam::new()
            //         .dest(rect.point())
            //         .scale(rect.size())
            //         .color(Color::WHITE),
            // );
            // let rect = graphics::Rect::new(0.0, WINSIZE - 3.0, WINSIZE, 3.0);
            // canvas.draw(
            //     &graphics::Quad,
            //     graphics::DrawParam::new()
            //         .dest(rect.point())
            //         .scale(rect.size())
            //         .color(Color::WHITE),
            // );

            // Finish Drawing
            canvas.finish(ctx)?;
        }
        Ok(())
    }
}

pub fn main() -> GameResult {
    rayon::ThreadPoolBuilder::new()
        .num_threads(CPU_NUM as usize)
        .build_global()
        .unwrap();

    use ggez::conf;
    let cb = ggez::ContextBuilder::new("Vicsek", "Rob Smith")
        .window_setup(conf::WindowSetup::default().title("Vicsek Demo"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(WINSIZE + 710.0, WINSIZE));
    let (mut ctx, event_loop) = cb.build()?;
    let state = MainState::new(&mut ctx)?;
    event::run(ctx, event_loop, state)
}
