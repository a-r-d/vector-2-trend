import { KMeansResult } from 'ml-kmeans/lib/KMeansResult';
import { ClusterRankingGroup, DataPoint } from './types';
import { orderBy } from 'lodash';

export function euclideanDistance(point1: number[], point2: number[]): number {
  if (point1.length !== point2.length)
    throw new Error('Both points must have the same dimension!');

  let sum = 0;
  for (let i = 0; i < point1.length; i++) {
    sum += Math.pow(point1[i] - point2[i], 2);
  }

  return Math.sqrt(sum);
}

export function averageDistance(point: number[], points: number[][]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    sum += euclideanDistance(point, points[i]);
  }
  return sum / points.length;
}

/**
 * Returns the silhouette score for each embedding in the dataset
 */
export function silhouetteScore(
  embeddings: number[][],
  labels: number[],
): number[] {
  // Number of clusters
  const n_clusters = Math.max(...labels) + 1;

  // Separate embeddings into clusters
  const clusters: number[][][] = [];
  for (let i = 0; i < n_clusters; i++) {
    clusters.push([]); // Correctly initialize each cluster as a separate array
  }

  for (let i = 0; i < embeddings.length; i++) {
    clusters[labels[i]].push(embeddings[i]);
  }

  // Compute silhouette scores
  const scores: number[] = [];
  for (let i = 0; i < embeddings.length; i++) {
    // Calculate a
    const a = averageDistance(embeddings[i], clusters[labels[i]]);

    // Calculate b for all other clusters and find the minimum
    let b = Number.POSITIVE_INFINITY;
    for (let j = 0; j < n_clusters; j++) {
      if (j != labels[i] && clusters[j].length > 0) {
        const tmp_b = averageDistance(embeddings[i], clusters[j]);
        if (tmp_b < b) {
          b = tmp_b;
        }
      }
    }

    // Calculate silhouette score
    let s = 0;
    if (a < b) {
      s = 1 - a / b;
    } else if (a > b) {
      s = b / a - 1;
    }
    scores.push(s);
  }

  return scores;
}

export function createCustomClusterRankings(
  kmeansResult: KMeansResult,
  silhouetteScores: number[],
  data: DataPoint[],
): ClusterRankingGroup[] {
  const clusterRankings: ClusterRankingGroup[] = [];

  for (let i = 0; i < kmeansResult.clusters.length; i++) {
    const clusterId = kmeansResult.clusters[i];
    const silhouetteScore = silhouetteScores[i];
    const record = data[i];
    const existingCluster = clusterRankings.find(
      (c) => c.clusterId === clusterId,
    );

    if (existingCluster) {
      existingCluster.records.push(record);
      existingCluster.count++;
      existingCluster.silhouetteScores.push(silhouetteScore);
    } else {
      clusterRankings.push({
        clusterId,
        records: [record],
        count: 1,
        silhouetteScores: [silhouetteScore],
        customDensity: 0,
        avgSilhouetteScore: 0,
      });
    }
  }

  for (const cluster of clusterRankings) {
    cluster.avgSilhouetteScore =
      cluster.silhouetteScores.reduce((a, b) => a + b, 0) /
      cluster.silhouetteScores.length;

    const customDensityScore =
      cluster.avgSilhouetteScore * Math.log(cluster.count);
    cluster.customDensity = customDensityScore;
  }

  const sortedClusters = orderBy(
    clusterRankings,
    (c) => c.customDensity,
    'desc',
  );
  return sortedClusters;
}
