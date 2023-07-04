use rand::Rng;

fn linear_model(xs: &Vec<f64>, params: &Vec<f64>) -> Vec<f64> {
    xs.iter().map(|x| {
        x * params[1] + params[0]
    }).collect()
}

fn mse(xs: &Vec<f64>, ys: &Vec<f64>) -> f64 {
    let diffs: Vec<f64> = xs.into_iter().zip(ys).map(|(a, b)| a - b).collect();
    let squares: Vec<f64> = diffs.iter().map(|x| x.powi(2)).collect();
    
    squares.iter().copied().reduce(|a,b| a + b).expect("Could not calculate sum.")
}

fn optimize_fn(
    xs: &Vec<f64>, 
    ys: &Vec<f64>, 
    model: fn(xs: &Vec<f64>, params: &Vec<f64>) -> Vec<f64>, 
    params: &mut Vec<f64>, 
    loss: fn(xs: &Vec<f64>, ys: &Vec<f64>) -> f64
) {
    let mut best_params = params.clone();

    let mut best_loss = 1000000.0;
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let new_params: Vec<f64> = (0..params.len()).map(|_| rng.gen::<f64>() * 10.0).collect();

        let predictions = model(xs, &new_params);
        let current_loss = loss(&predictions, &ys);

        if current_loss < best_loss {
            println!("Found loss: {:?} previous best: {:?}", current_loss, best_loss);
            best_loss = current_loss;
            best_params = new_params.clone();
        }
    }
    params.clone_from_slice(&best_params);
    //println!("{:?} {:?}", best_params, best_loss);
    //let predictions = model(xs, &best_params);
    //println!("Predictions: {:?}", predictions);
}

fn main() {
    let data = vec![0.1, 0.5, 0.3];
    let labels = vec![0.2, 1.0, 0.6];
    
    let mut params = vec![0.0, 1.0];
    // Modeling
    let out = linear_model(&data, &params);

    println!("{:?}", out);
    println!("Model error: {:?}", mse(&out, &labels));

    optimize_fn(&data, &labels, linear_model, &mut params, mse);

    let out = linear_model(&data, &params);

    println!("{:?}", out);
    println!("Model error after optimization: {:?}", mse(&out, &labels));

    println!("{:?} {:?} {:?} {:?}", data, out, labels, params);
}
