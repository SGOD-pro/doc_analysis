// ── Document ──────────────────────────────────────────────
export interface Document {
  doc_id: string;
  document_name: string;
  original_name?: string;
  format: string;
  total_pages: number;
  status?: "uploaded" | "processing" | "processed" | "error";
  is_processed?: boolean;
  error?: string;
}

export interface PageData {
  page: number;
  text: string;
  paragraphs: string[];
  word_count: number;
  char_count: number;
  heading?: string;
}

export interface ProcessedDocument {
  doc_id: string;
  document_name: string;
  format: string;
  total_pages: number;
  pages: PageData[];
}

// ── Entity ────────────────────────────────────────────────
export interface Entity {
  entity: string;
  type: string;
  type_label: string;
  frequency: number;
}

export interface EntityData {
  doc_id: string;
  document_name: string;
  total_entities: number;
  unique_entities: number;
  entity_types_found: string[];
  entities: Entity[];
  by_type: Record<string, Entity[]>;
  top_entities: Entity[];
}

export interface AggregatedEntities {
  total_documents_analyzed: number;
  total_unique_entities: number;
  total_entity_mentions: number;
  top_entities: Entity[];
  by_type: Record<string, Entity[]>;
  top_organizations: Entity[];
  top_people: Entity[];
  top_locations: Entity[];
  top_products: Entity[];
}

// ── Topic ─────────────────────────────────────────────────
export interface Topic {
  topic_id: number;
  topic: string;
  keywords: string[];
  document_count?: number;
  frequency?: number;
  source_doc?: string;
}

export interface Keyword {
  keyword: string;
  count: number;
  score?: number;
}

export interface TopicData {
  doc_id: string;
  document_name: string;
  total_topics: number;
  topics: Topic[];
  top_keywords: Keyword[];
  page_trends: { page: number; top_keywords: Keyword[] }[];
  bertopic_used: boolean;
}

export interface AggregatedTopics {
  total_documents_analyzed: number;
  total_topics: number;
  top_keywords: Keyword[];
  topic_distribution: { topic: string; count: number }[];
  all_topics: Topic[];
}

// ── Graph ─────────────────────────────────────────────────
export interface GraphNode {
  id: string;
  label: string;
  type: string;
  type_label: string;
  frequency: number;
  degree: number;
  centrality: number;
  betweenness: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relationship: string;
  weight: number;
}

export interface GraphAnalytics {
  total_nodes: number;
  total_edges: number;
  density: number;
  most_connected_entity: string;
  average_degree: number;
}

export interface GraphData {
  doc_id?: string;
  document_name?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  analytics: GraphAnalytics;
}

// ── Search ────────────────────────────────────────────────
export interface SearchResult {
  doc_id: string;
  document_name: string;
  page: number;
  paragraph_idx: number;
  snippet: string;
  relevance_score: number;
  matched_terms: string[];
}

export interface SearchResponse {
  query: string;
  total_results: number;
  results: SearchResult[];
}

// ── Analytics ─────────────────────────────────────────────
export interface AnalyticsSummary {
  total_documents: number;
  total_pages: number;
  total_entities: number;
  total_topics: number;
  total_graph_nodes: number;
  total_graph_edges: number;
  average_graph_density: number;
  most_connected_entities: string[];
}

export interface Analytics {
  summary: AnalyticsSummary;
  documents: Document[];
  document_format_distribution: { format: string; count: number }[];
  entity_analytics: {
    top_organizations: Entity[];
    top_people: Entity[];
    top_locations: Entity[];
    top_products: Entity[];
    entity_type_distribution: { type: string; count: number }[];
  };
  topic_analytics: {
    top_keywords: Keyword[];
    topic_distribution: { topic: string; count: number }[];
  };
  graph_analytics: GraphAnalytics & { most_connected_entities: string[] };
}

// ── Upload ────────────────────────────────────────────────
export interface UploadResponse {
  success: boolean;
  doc_id: string;
  filename: string;
  format: string;
  message: string;
}

export interface ProcessResponse {
  success: boolean;
  doc_id: string;
  document_name: string;
  total_pages: number;
  entities_found: number;
  topics_found: number;
  graph_nodes: number;
  graph_edges: number;
  status: string;
}
