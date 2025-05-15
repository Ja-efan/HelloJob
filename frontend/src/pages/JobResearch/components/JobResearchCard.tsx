import { FaRegBookmark, FaBookmark } from "react-icons/fa";

interface JobResearchCardProps {
  onClick: () => void;
  jobRoleName: string;
  jobRoleAnalysisTitle: string;
  jobRoleCategory: string;
  jobRoleViewCount: number;
  jobRoleBookmarkCount: number;
  bookmark: boolean;
  updatedAt: string;
}

function JobResearchCard({
  onClick,
  jobRoleName,
  jobRoleAnalysisTitle,
  jobRoleCategory,
  jobRoleViewCount,
  jobRoleBookmarkCount,
  bookmark,
  updatedAt,
}: JobResearchCardProps) {
  return (
    <div
      onClick={onClick}
      className="w-[800px] h-[110px] border-l-4 border-[#6F4BFF] bg-white rounded-lg rounded-l-xs cursor-pointer shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between p-4"
    >
      <div className="flex justify-between items-start">
        <div className="flex flex-col">
          <div className="flex items-center gap-4">
            <h3 className="text-lg font-bold text-[#333] mb-1">
              {jobRoleAnalysisTitle}
            </h3>
            <span className="text-sm font-medium text-[#6F4BFF]">
              {jobRoleCategory}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-[#edeafb] text-[#6F4BFF] text-sm px-2 py-0.5 rounded-full">
              {jobRoleName}
            </span>
          </div>
        </div>
        <div className="flex items-center">
          {bookmark ? (
            <FaBookmark className="text-[#6F52E0]" />
          ) : (
            <FaRegBookmark />
          )}
        </div>
      </div>

      <div className="flex justify-between items-center mt-2">
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span className="flex items-center gap-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
            {jobRoleViewCount}
          </span>
          <span className="flex items-center gap-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
              />
            </svg>
            {jobRoleBookmarkCount}
          </span>
        </div>
        <span className="text-xs text-gray-400">
          {new Date(updatedAt).toLocaleDateString("ko-KR", {
            year: "numeric",
            month: "long",
            day: "numeric",
          })}
        </span>
      </div>
    </div>
  );
}

export default JobResearchCard;
