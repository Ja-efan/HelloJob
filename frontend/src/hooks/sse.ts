import { useEffect } from "react";
import { toast } from "sonner";
import { useNavigate } from "react-router";

const baseURL = import.meta.env.DEV ? "" : "https://k12b105.p.ssafy.io";

export default function useSSE(isLoggedIn: boolean) {
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoggedIn) return;

    const eventSource = new EventSource(`${baseURL}/api/v1/sse/subscribe`);

    // event 이름이 없는 일반 메시지
    eventSource.onmessage = (e: MessageEvent) => {
      console.log("📨 일반 메시지:", e.data);
    };

    // 커스텀 이벤트 수신
    eventSource.addEventListener(
      "company-analysis-completed",
      (e: MessageEvent) => {
        const companyAnalysisId = JSON.parse(e.data);
        toast("기업 분석이 완료되었습니다!", {
          description: "결과를 확인하려면 클릭하세요",
          action: {
            label: "보러가기",
            onClick: () =>
              navigate(`/corporate-research?openId=${companyAnalysisId}`),
          },
        });
      }
    );

    eventSource.onerror = (err) => {
      console.error("SSE 오류:", err);
    };

    return () => {
      eventSource.close();
    };
  }, [isLoggedIn]);
}
