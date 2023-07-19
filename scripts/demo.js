const Vector2Trends = require('../dist/index');
const fs = require('fs');

const testData = fs.readFileSync('./src/sample-data.json', 'utf8');
const testDataPoints = JSON.parse(testData);

const clusters = Vector2Trends.default.cluster({
  records: testDataPoints,
  n: 1536,
  pcaDimensions: 20,
  clusteringAlgorithm: 'kmeans',
});

console.log(clusters);

const classify = async () => {
  const calssifications = await Vector2Trends.default.classify({
    clusteringResult: clusters,
    // TODO: change this to your own API key
    openAiApiKey: 'sk-xxxxxxxxxxxxxxxxxxx',
  });

  console.log(calssifications);

  // topics:
  console.log(calssifications.map(x => ({ topic: x.name, score: x.score })))
  return calssifications;
};

classify();
