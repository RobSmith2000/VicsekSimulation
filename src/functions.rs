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
use ggez::{
    graphics::Rect,
    input::{
        self,
        keyboard::{self, KeyInput},
    },
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    time::Instant,
};

pub const PI: f32 = std::f32::consts::PI;
pub const WINSIZE: f32 = 900.0;
pub const CPU_NUM: i32 = 8;

pub fn randpartigen(boxlen: f32) -> Vec<f32> {
    // Generate a random position and bearning inside the box
    let mut rng: ThreadRng = rand::thread_rng();
    let coords: Vec<f32> = vec![
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(boxlen)) as f32,
        rng.gen_range((0.0)..(2. * PI)) as f32,
    ];
    return coords;
}

pub fn box_nn(boxcoord: Vec<i32>, boxlen: f32) -> Vec<Vec<i32>> {
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

pub fn new_angle_for_index_array(
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

pub fn convert_hsv_to_rgb(pixel: &Vec<f32>) -> [f32; 3] {
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

pub fn semicirclearc(angle: f32, boxlen: f32) -> [Point2<f32>; 361] {
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
pub fn RangeQuery(
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

pub fn DBSCAN(
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

    // return vec![(1, vec![3 as usize])];
    return clustermap;
}

pub fn comcalc(indecieslist: &Vec<usize>, partilist: &Vec<Vec<f32>>, boxlen: f32) -> Vec<f32> {
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

pub fn normvel_cluster(values: &Vec<usize>, partilist: &Vec<Vec<f32>>) -> f32 {
    let mut sumsin: f32 = 0.0;
    let mut sumcos: f32 = 0.0;
    for value in values {
        let parti = &partilist[*value];
        sumsin += &parti[2].sin();
        sumcos += &parti[2].cos();
    }
    let normvel = (1.0 / partilist.len() as f32) * f32::sqrt((sumsin * sumsin) + (sumcos * sumcos));
    return normvel;
}

pub fn normangvel_cluster(values: &Vec<usize>, partilist: &Vec<Vec<f32>>, boxlen: f32) -> f32 {
    let com = comcalc(values, &partilist, boxlen);
    let xcom = com[0];
    let ycom = com[1];
    let mut sum = 0.;
    for value in values {
        let parti = &partilist[*value];

        // TODO refactor this as a torus distance function
        let partix: f32 = parti[0];
        let partiy: f32 = parti[1];
        let partia: f32 = parti[2];

        let velx: f32 = partia.sin();
        let vely: f32 = partia.cos();
        let mut distx: f32 = partix - xcom;
        let mut disty: f32 = partiy - ycom;

        // 3x3 box coord correction with periodic boundry conditions
        if (distx).abs() > boxlen / 2. {
            if partix > xcom {
                distx -= boxlen;
            } else {
                distx += boxlen;
            }
        }
        if (disty).abs() > boxlen / 2. {
            if partiy > ycom {
                disty -= boxlen;
            } else {
                disty += boxlen;
            }
        }
        let distsquared: f32 = (distx * distx) + (disty * disty);

        let angmom = (((distx * vely) - (disty * velx)).abs()) / distsquared.sqrt();
        sum += angmom;
    }
    let normangvel = sum / values.len() as f32;
    return normangvel;
}

pub fn disttorus(parti1: &Vec<f32>, parti2: &Vec<f32>, boxlen: f32) -> f32 {
    let mut dx: f32 = (parti1[0] - parti2[0]).abs();
    let mut dy: f32 = (parti1[1] - parti2[1]).abs();

    if dx > boxlen / 2.0 {
        dx = boxlen - dx;
    }
    if dy > boxlen / 2.0 {
        dy = boxlen - dy;
    }

    let distsquared: f32 = dx * dx + dy * dy;
    return distsquared.sqrt();
}
pub fn threepointcirc(
    partia: &Vec<f32>,
    partib: &Vec<f32>,
    partic: &Vec<f32>,
    boxlen: f32,
) -> Vec<f32> {
    let mut xDelta_a = partia[0] - partic[0];
    let mut yDelta_a = partia[1] - partic[1];
    let mut xDelta_b = partib[0] - partic[0];
    let mut yDelta_b = partib[1] - partic[1];

    // changing mid point for periodic boundry conditions
    if xDelta_a > boxlen / 2. {
        xDelta_a -= boxlen;
    }
    if xDelta_a < -boxlen / 2. {
        xDelta_a += boxlen;
    }
    if yDelta_a > boxlen / 2. {
        yDelta_a -= boxlen;
    }
    if yDelta_a < -boxlen / 2. {
        yDelta_a += boxlen;
    }

    if xDelta_b > boxlen / 2. {
        xDelta_b -= boxlen;
    }
    if xDelta_b < -boxlen / 2. {
        xDelta_b += boxlen;
    }
    if yDelta_b > boxlen / 2. {
        yDelta_b -= boxlen;
    }
    if yDelta_b < -boxlen / 2. {
        yDelta_b += boxlen;
    }

    let acx = partic[0] + (xDelta_a / 2.);
    let acy = partic[1] + (yDelta_a / 2.);
    let bcx = partic[0] + (xDelta_b / 2.);
    let bcy = partic[1] + (yDelta_b / 2.);

    let aSlope = yDelta_a / xDelta_a;
    let bSlope = yDelta_b / xDelta_b;

    let mut centerx =
        ((aSlope * bSlope) / (aSlope - bSlope)) * ((bcy + (bcx / bSlope)) - (acy + (acx / aSlope)));

    let mut centery = (-centerx / (aSlope)) + (acy + (acx / aSlope));

    if centerx < 0. {
        centerx += boxlen;
    }
    if centerx > boxlen {
        centerx -= boxlen;
    }
    if centery < 0. {
        centery += boxlen;
    }
    if centery > boxlen {
        centery -= boxlen;
    }
    let mut cor = vec![-1., -1.];
    if (disttorus(partia, &vec![centerx, centery], boxlen) < boxlen / 10.)
        && (disttorus(partib, &vec![centerx, centery], boxlen) < boxlen / 10.)
        && (disttorus(partic, &vec![centerx, centery], boxlen) < boxlen / 10.)
    {
        cor = vec![centerx, centery];
    }
    return cor;
}
