"use client";

import { useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type SearchResult = {
  doc_id: string;
  chunk_index: number;
  content: string;
  metadata: { question?: string; source?: string; [key: string]: unknown };
  distance: number;
};

function sourceBadge(source: unknown): { label: string; className: string } {
  if (typeof source === "string" && source.startsWith("aihub")) {
    return {
      label: "AI-Hub",
      className:
        "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
    };
  }
  return {
    label: "HuggingFace",
    className:
      "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  };
}

type AskResponse = {
  answer: string;
  sources: SearchResult[];
};

function renderContent(content: string) {
  const idx = content.indexOf("답변:");
  if (idx <= 0) return content;
  return (
    <>
      {content.slice(0, idx).trim()}
      <br />
      {content.slice(idx)}
    </>
  );
}

function SearchColumn({
  model,
  label,
}: {
  model: "e5" | "bge_m3";
  label: string;
}) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5, model }),
      });
      if (!res.ok) throw new Error(`요청 실패 (${res.status})`);
      const data: AskResponse = await res.json();
      setAnswer(data.answer);
      setResults(data.sources);
    } catch {
      setError("검색에 실패했습니다. 백엔드 서버(FastAPI)가 켜져 있는지 확인하세요.");
      setAnswer(null);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col gap-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
        {label}
      </p>

      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="궁금한 내용을 입력하세요"
          className="flex-1 rounded-md border border-zinc-300 bg-white px-4 py-2 text-black outline-none focus:border-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-black px-5 py-2 font-medium text-white disabled:opacity-50 dark:bg-white dark:text-black"
        >
          {loading ? "검색 중..." : "검색"}
        </button>
      </form>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {answer && (
        <div className="rounded-lg border border-zinc-300 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
          <p className="mb-1 text-xs font-medium text-zinc-500">답변</p>
          <p className="whitespace-pre-wrap text-sm text-black dark:text-zinc-50">
            {answer}
          </p>
        </div>
      )}

      <ul className="flex flex-col gap-3">
        {results.map((r) => {
          const badge = sourceBadge(r.metadata?.source);
          return (
            <li
              key={`${r.doc_id}-${r.chunk_index}`}
              className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
            >
              <div className="mb-1 flex items-center gap-2">
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${badge.className}`}
                >
                  {badge.label}
                </span>
                {r.metadata?.question && (
                  <p className="text-xs font-medium text-zinc-500">
                    원본 질문: {String(r.metadata.question)}
                  </p>
                )}
              </div>
              <p className="text-sm text-black dark:text-zinc-50">{renderContent(r.content)}</p>
              <p className="mt-2 text-xs text-zinc-400">
                distance: {r.distance.toFixed(4)}
              </p>
            </li>
          );
        })}
      </ul>

      {!loading && !error && results.length === 0 && (
        <p className="text-sm text-zinc-400">검색 결과가 여기에 표시됩니다.</p>
      )}
    </div>
  );
}

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex w-full max-w-5xl flex-col gap-6 px-6 py-16">
        <div>
          <h1 className="text-2xl font-semibold text-black dark:text-zinc-50">
            RAG 검색 데모 — 임베딩 모델 비교
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            좌우에 각각 질문을 입력해서 e5-base와 bge-m3 검색 결과를 비교해보세요
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 md:divide-x md:divide-zinc-200 dark:md:divide-zinc-800">
          <div className="md:pr-8">
            <SearchColumn model="e5" label="e5-base" />
          </div>
          <div className="md:pl-8">
            <SearchColumn model="bge_m3" label="bge-m3" />
          </div>
        </div>
      </main>
    </div>
  );
}
