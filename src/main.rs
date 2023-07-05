use rand::Rng;

fn linear_model(xs: &Vec<f64>, params: &Vec<f64>) -> Vec<f64> {
    xs.iter().map(|x| {
        x * params[1] + params[0]
    }).collect()
}

fn mse(xs: &Vec<f64>, ys: &Vec<f64>) -> f64 {
    let diffs: Vec<f64> = xs.into_iter().zip(ys).map(|(a, b)| a - b).collect();
    let squared: Vec<f64> = diffs.iter().map(|x| x.powi(2)).collect();
    
    squared.iter().copied().reduce(|a,b| a + b).expect("Could not calculate sum.")
}

fn random_optimize_fn(
    xs: &Vec<f64>, 
    ys: &Vec<f64>, 
    model: fn(xs: &Vec<f64>, params: &Vec<f64>) -> Vec<f64>, 
    params: &mut Vec<f64>, 
    loss: fn(xs: &Vec<f64>, ys: &Vec<f64>) -> f64
) {
    let mut best_params = params.clone();

    let mut best_loss = std::f64::MAX;
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
}

fn naive_gradient_descent_optimize_fn(
    xs: &Vec<f64>, 
    ys: &Vec<f64>, 
    model: fn(xs: &Vec<f64>, params: &Vec<f64>) -> Vec<f64>, 
    params: &mut Vec<f64>, 
    loss: fn(xs: &Vec<f64>, ys: &Vec<f64>) -> f64,
) {
    let mut best_params = params.clone();
    let h = 0.0000001;
    let lr = 0.001;

    for _ in 0..100000 {
        let mut gradient = vec![0.0; best_params.len()];
        
        // Calulate gradients
        for i in 0..best_params.len() {
            let mut temp_params = best_params.clone();
            temp_params[i] = temp_params[i] + h;
            gradient[i] = (
                loss(&model(xs, &temp_params), &ys) - 
                loss(&model(xs, &best_params), &ys)
            ) / h;
        }

        // Update parameters
        for i in 0..best_params.len() {
            best_params[i] = best_params[i] - lr * gradient[i];
        }

    }    
    params.clone_from_slice(&best_params);
}

fn main() {
    let data = vec![0.1, 0.5, 0.3];
    let labels = vec![0.2, 1.0, 0.6];
    
    let mut params = vec![0.0, 1.0];
    // Modeling
    let out = linear_model(&data, &params);

    println!("{:?}", out);
    println!("Model error: {:?}", mse(&out, &labels));

    naive_gradient_descent_optimize_fn(&data, &labels, linear_model, &mut params, mse);

    let out = linear_model(&data, &params);

    println!("{:?}", out);
    println!("Model error after optimization: {:?}", mse(&out, &labels));

    println!("{:?} {:?} {:?} {:?}", data, out, labels, params);
}
