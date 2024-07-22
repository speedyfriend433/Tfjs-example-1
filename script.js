import * as tf from '@tensorflow/tfjs';

const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
});

    model.fit(xs, ys, {epochs: 200}).then(() => {
    model.predict(tf.tensor2d([5], [1, 1])).print();
    model.predict(tf.tensor2d([6], [1, 1])).print();
    drawChart();
});

async function drawChart() {
    const predictions = model.predict(tf.tensor2d([1, 2, 3, 4, 5, 6], [6, 1]));
    const preds = await predictions.array();

    const ctx = document.getElementById('myChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [1, 2, 3, 4, 5, 6],
            datasets: [{
                label: 'Predicted values',
                data: preds.map(p => p[0]),
                borderColor: 'blue',
                fill: false
            }, {
                label: 'Original values',
                data: [1, 3, 5, 7],
                borderColor: 'red',
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
}
