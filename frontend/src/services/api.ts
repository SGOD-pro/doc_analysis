import axios from "axios";
import type {
  AggregatedEntities,
  AggregatedTopics,
  Analytics,
  EntityData,
  GraphData,
  ProcessResponse,
  SearchResponse,
  TopicData,
  UploadResponse,
} from "../types";

const api = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

// ── Documents ──────────────────────────────────────────────
export const uploadDocument = async (file: File): Promise<UploadResponse> => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const processDocument = async (docId: string): Promise<ProcessResponse> => {
  const { data } = await api.post<ProcessResponse>(`/process/${docId}`);
  return data;
};

export const listDocuments = async () => {
  const { data } = await api.get("/documents");
  return data;
};

export const getDocument = async (docId: string) => {
  const { data } = await api.get(`/documents/${docId}`);
  return data;
};

export const deleteDocument = async (docId: string) => {
  const { data } = await api.delete(`/documents/${docId}`);
  return data;
};

// ── Search ─────────────────────────────────────────────────
export const searchDocuments = async (
  query: string,
  limit = 20
): Promise<SearchResponse> => {
  const { data } = await api.get<SearchResponse>("/search", {
    params: { q: query, limit },
  });
  return data;
};

// ── Entities ───────────────────────────────────────────────
export const getAllEntities = async (): Promise<AggregatedEntities> => {
  const { data } = await api.get<AggregatedEntities>("/entities");
  return data;
};

export const getDocEntities = async (docId: string): Promise<EntityData> => {
  const { data } = await api.get<EntityData>(`/entities/${docId}`);
  return data;
};

// ── Topics ─────────────────────────────────────────────────
export const getAllTopics = async (): Promise<AggregatedTopics> => {
  const { data } = await api.get<AggregatedTopics>("/topics");
  return data;
};

export const getDocTopics = async (docId: string): Promise<TopicData> => {
  const { data } = await api.get<TopicData>(`/topics/${docId}`);
  return data;
};

// ── Knowledge Graph ────────────────────────────────────────
export const getKnowledgeGraph = async (): Promise<GraphData> => {
  const { data } = await api.get<GraphData>("/knowledge-graph");
  return data;
};

export const getDocGraph = async (docId: string): Promise<GraphData> => {
  const { data } = await api.get<GraphData>(`/knowledge-graph/${docId}`);
  return data;
};

// ── Analytics ──────────────────────────────────────────────
export const getAnalytics = async (): Promise<Analytics> => {
  const { data } = await api.get<Analytics>("/analytics");
  return data;
};

export default api;
