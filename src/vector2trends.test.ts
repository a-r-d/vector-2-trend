import { ClusteringArgument, DataPoint } from './types';
import { Vector2Trend } from './vector2Trend';
import fs from 'fs';

const testData = fs.readFileSync('./src/sample-data.json', 'utf8');

const testDataPoints = JSON.parse(testData) as DataPoint[];

describe('Vector2Trend', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should perform clustering', () => {
    // Setup
    const clusteringArgs: ClusteringArgument = {
      records: testDataPoints,
      n: 1536,
      pcaDimensions: 20,
      clusteringAlgorithm: 'kmeans',
    };

    const clusterResult = Vector2Trend.cluster(clusteringArgs);

    expect(clusterResult.rankings.length).toBeLessThan(100);
    expect(clusterResult.rankings[0]).toEqual({
      avgSilhouetteScore: expect.any(Number),
      clusterId: expect.any(Number),
      count: expect.any(Number),
      customDensity: expect.any(Number),
      records: expect.any(Array),
      silhouetteScores: expect.any(Array),
    });

    /**
     * The strongest cluster in this group is 
     * people complaining they did not get their order
     */
    const allTextFirstCluster = clusterResult.rankings[0].records.map(r => r.text);
    expect(allTextFirstCluster).toContain(`i didn’t receive my order`);
  });
});
