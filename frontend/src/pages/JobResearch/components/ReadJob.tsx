import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FaRegBookmark, FaBookmark } from "react-icons/fa";
import { useState, useEffect } from "react";
import { jobRoleAnalysis } from "@/api/jobRoleAnalysisApi";
import { getJobRoleDetail } from "@/types/jobResearch";
import { Button } from "@/components/Button";

interface ReadJobProps {
  onClose: () => void;
  id: number;
}

function ReadJob({ onClose, id }: ReadJobProps) {
  const queryClient = useQueryClient();
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [isValidId, setIsValidId] = useState(false);
  // id 유효성 검사
  useEffect(() => {
    if (Number.isInteger(id) && id > 0) {
      setIsValidId(true);
    } else {
      setIsValidId(false);
    }
  }, [id]);

  // 직무 분석 레포트 상세 정보 불러오기
  const { data: jobDetail, isLoading } = useQuery({
    queryKey: ["jobRoleDetail", id],
    queryFn: async () => {
      const response = await jobRoleAnalysis.getJobDetail(id);
      return response.data as getJobRoleDetail;
    },
    enabled: isValidId,
  });

  // jobDetail이이 변경될 때 북마크 상태 업데이트
  useEffect(() => {
    if (jobDetail) {
      setIsBookmarked(jobDetail.bookmark);
    }
  }, [jobDetail]);

  // 북마크 추가 mutation
  const addJobBookmarkMutation = useMutation({
    mutationFn: () => jobRoleAnalysis.postBookmark({ jobRoleAnalysisId: id }),
    onSuccess: () => {
      console.log("북마크 추가 성공");
      setIsBookmarked(true);
      queryClient.invalidateQueries({
        queryKey: ["jobRoleDetail", id],
      });
      queryClient.invalidateQueries({ queryKey: ["jobRoleList"] });
    },
  });

  // 북마크 삭제 mutation
  const removeJobBookmarkMutation = useMutation({
    mutationFn: () => jobRoleAnalysis.deleteBookmark(id),
    onSuccess: () => {
      console.log("북마크 삭제 성공");
      setIsBookmarked(false);
      queryClient.invalidateQueries({
        queryKey: ["jobRoleDetail", id],
      });
      queryClient.invalidateQueries({ queryKey: ["jobRoleList"] });
    },
  });
  // 북마크 토글 핸들러
  const toggleBookmark = () => {
    if (isBookmarked) {
      addJobBookmarkMutation.mutate();
    } else {
      removeJobBookmarkMutation.mutate();
    }
  };

  return (
    <div className="h-[90vh] w-[940px] bg-white rounded-t-xl py-8 px-12 overflow-y-auto">
      <header className="flex w-full justify-between items-end mb-4">
        <h1 className="text-2xl font-bold">
          {isLoading || !jobDetail ? "로딩 중..." : jobDetail.companyName} 기업
          분석 -{" "}
          {isLoading || !jobDetail
            ? ""
            : new Date(jobDetail.createdAt).toLocaleDateString("ko-KR")}
        </h1>
        <button
          onClick={toggleBookmark}
          className="flex cursor-pointer items-center gap-1 px-3 py-2 rounded-md hover:bg-gray-100 transition-colors border border-gray-200"
        >
          {isBookmarked ? (
            <FaBookmark className="text-[#6F52E0]" />
          ) : (
            <FaRegBookmark />
          )}
          <span>북마크</span>
        </button>
      </header>
      <Button onClick={onClose}>닫기</Button>
    </div>
  );
}

export default ReadJob;
