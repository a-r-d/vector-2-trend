import {
  ClassificationRequest,
  ClassifiedClusterResponse,
  ClusteringArgument,
  ClusteringResult,
  DataPoint,
} from './types';
import { first, trim } from 'lodash';
import {
  Configuration,
  CreateChatCompletionResponseChoicesInner,
  OpenAIApi,
} from 'openai';
import { kmeans } from 'ml-kmeans';
import { PCA } from 'ml-pca';
import { createCustomClusterRankings, silhouetteScore } from './utils';
import { Logger } from 'tslog';

const debugLogging = () => {
  return new Logger({
    name: 'vector2trend',
    minLevel: 2,
    type: 'pretty',
  });
};

const defaultLogging = () => {
  return new Logger({
    name: 'vector2trend',
    minLevel: 5,
    type: 'pretty',
  });
};

let log = defaultLogging();

export const cleanClassification = <T>(
  classification: CreateChatCompletionResponseChoicesInner[],
): T | null => {
  try {
    const textResponse = first(classification)?.message?.content;
    if (!textResponse) {
      log.debug('No text response from classification', classification);
      return null;
    }
    const trimmedString = trim(
      textResponse.substring(textResponse.indexOf('{')),
    );

    log.debug('Parsing classification...', trimmedString);
    const parsed = JSON.parse(trimmedString);

    log.debug('Parsed classification', parsed);
    log.debug('Parsed formatted', JSON.stringify(parsed, null, 2));
    return parsed;
  } catch (error) {
    log.debug('Error parsing classification', error);
    return null;
  }
};

export const callCompletion = async ({
  apiKey,
  promptText,
  temperature,
}: {
  apiKey: string;
  promptText: string;
  temperature?: number;
}) => {
  const configuration = new Configuration({
    apiKey: apiKey,
  });
  const openai = new OpenAIApi(configuration);
  const response = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: promptText }],
    max_tokens: 1000,
    temperature: temperature ?? 0.5,
  });

  return response.data.choices;
};

/**
 * This will always sort the clusters by size and order by the top 10
 *
 * This is not necessarily the best way to do this you might want the
 * tightest clusters but this is a good starting point
 */
export const classifyClusters = async (
  args: ClassificationRequest,
): Promise<ClassifiedClusterResponse[]> => {
  const nTopics = args.nTopics ?? 10;
  if (nTopics > args.clusteringResult.rankings.length) {
    throw new Error(
      `Number of topics cannot exceed number of clustering results. nTopics=${nTopics}`,
    );
  }

  /**
   * Assume these are sorted descending by score
   */
  // Only get the top n groups
  const topnGroups = args.clusteringResult.rankings.slice(0, nTopics);

  // this is here for sanity so you don't burn a TON of tokens
  const maxElementsPerGroup = args.elementsPerGroup ?? 10;

  const topNGroupsTruncatedMembers = topnGroups.map((group) => {
    return {
      ...group,
      records: group.records.slice(0, maxElementsPerGroup),
    };
  });

  const prompt = `You will be asked to summarize the topics of the following groups text. Please provide a short name for each of groups between 1-6 words based on the content.
  You must return a single JSON formatted list of strings (Array<string>) with the same length as the number of groups. There are ${
    topNGroupsTruncatedMembers.length
  } groups.
  ${topNGroupsTruncatedMembers
    .map((cluster, i) => {
      return `Group ${i + 1}:
  ${cluster.records
    .map((record, i) => `Feedback ${i + 1}: ${record.text}`)
    .join('\n')}
  `;
    })
    .join('\n')})
  }
  JSON Response:`;
  const completion = await callCompletion({
    apiKey: args.openAiApiKey,
    promptText: prompt,
    temperature: args.temperature ?? 0.1,
  });
  const resultClassification = cleanClassification<string[]>(completion);

  if (resultClassification) {
    return topnGroups.map((cluster, i) => {
      return {
        ...cluster,
        score: cluster.customDensity,
        name: resultClassification[i] ?? 'N/A',
      };
    });
  }

  return topnGroups.map((cluster) => {
    return {
      ...cluster,
      score: cluster.customDensity,
      name: 'N/A',
    };
  });
};

export class Vector2Trend {
  static cluster(args: ClusteringArgument): ClusteringResult {
    if (args.records.length < args.pcaDimensions) {
      log.debug(
        'Not enough valid vectors to run kmeans',
        args.records.length,
        args.pcaDimensions,
      );
      throw new Error('Not enough valid vectors to run kmeans');
    }

    const originalDatapoints = args.records;
    /**
     * Do PCA to reduce dimensionality
     */
    const pca = new PCA(originalDatapoints.map((row) => row.vector));
    const reducedDimensions = pca.predict(
      originalDatapoints.map((row) => row.vector),
      {
        // You cannot have more components and the number of the records
        nComponents: args.pcaDimensions,
      },
    );

    const dataPointsReduced: DataPoint[] = [];
    for (let i = 0; i < originalDatapoints.length; i++) {
      dataPointsReduced.push({
        id: originalDatapoints[i].id,
        text: originalDatapoints[i].text,
        vector: reducedDimensions.getRow(i),
      });
    }

    // Convert embeddings back into a list format
    const embeddings = dataPointsReduced.map((row) => row.vector);

    // Decide on the number of clusters (you may need to adjust this)
    // this algo feels about correct
    const num_clusters = Math.ceil(originalDatapoints.length / 4) + 1;

    // Perform k-means clustering
    const kmeansResult = kmeans(embeddings, num_clusters, {
      // TODO ??
      initialization: 'kmeans++',
      seed: 42,
    });

    const silhouetteScores = silhouetteScore(
      dataPointsReduced.map((x) => x.vector),
      kmeansResult.clusters,
    );

    const rankings = createCustomClusterRankings(
      kmeansResult,
      silhouetteScores,
      dataPointsReduced,
    );

    return {
      rankings: rankings,
    };
  }

  static async classify(
    args: ClassificationRequest,
  ): Promise<ClassifiedClusterResponse[]> {
    return classifyClusters(args);
  }

  static enableLogging() {
    log = debugLogging();
  }
}
