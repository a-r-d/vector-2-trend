export interface DataPoint {
  id: number | string;
  text: string;
  records: number[];
}

export type ClusteringArgument = {
  records: DataPoint[];
  n: number;
  pcaDimensions: number;
  // we will add more in the future
  clusteringAlgorithm: 'kmeans';
};

export type ClusterRankingGroup = {
  silhouetteScores: number[];
  avgSilhouetteScore: number;
  clusterId: number;
  records: DataPoint[];
  count: number;
  customDensity: number;
};

export type ClusteringResult = {
  rankings: ClusterRankingGroup[];
};

export type ClassificationRequest = {
  clusteringResult: ClusteringResult;
  openAiApiKey: string;

  nTopics?: number // defaults to 10
};

export type ClassifiedClusterResponse = {
  // this is the density score
  score: number;

  // these are the same as from the cluster ranking result
  clusterId: number;
  count: number;
  records: DataPoint[];

  // generated from GPT
  name: string;
};
