import { FaRegBookmark, FaBookmark } from "react-icons/fa";

interface CorporateReportCardProps {
  onClick: () => void;
  companyName: string;
  createdAt: string;
  companyViewCount: number;
  companyLocation: string;
  companyAnalysisBookmarkCount: number;
  bookmark: boolean;
  dartCategory?: string[];
}

function CorporateReportCard({
  onClick,
  companyName,
  createdAt,
  companyViewCount,
  companyLocation,
  companyAnalysisBookmarkCount,
  bookmark,
  dartCategory,
}: CorporateReportCardProps) {
  return (
    <div
      onClick={onClick}
      className="w-[210px] h-[180px] bg-white rounded-lg cursor-pointer border border-gray-200 p-3 hover:shadow-sm transition-shadow flex flex-col justify-between"
    >
      <div>
        <div className="flex items-center w-full border-b border-[#AF9BFF]/80 pb-1 justify-between">
          <h3 className="text-base font-bold text-gray-800 truncate">
            {companyName}
          </h3>
          {bookmark ? (
            <FaBookmark className="text-[#6F52E0]" />
          ) : (
            <FaRegBookmark />
          )}
        </div>
        <p className="text-sm text-gray-600 mt-1 truncate">{companyLocation}</p>
        <p className="text-sm text-gray-500 mt-1">
          {new Date(createdAt).toLocaleDateString()}
        </p>

        <div className="flex flex-wrap gap-1 mt-2 min-h-[40px]">
          {dartCategory?.map((category, index) => (
            <span
              key={index}
              className="text-[10px] bg-purple-100 text-purple-800 rounded-full px-2 py-0.5"
            >
              {category}
            </span>
          ))}
        </div>

        <div className="flex items-center justify-end text-sm text-gray-600 mt-2">
          <span className="flex items-center">
            <svg
              className="w-3 h-3 mr-1"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
              <path
                fillRule="evenodd"
                d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            {companyViewCount}
          </span>
          <span className="flex items-center ml-2">
            <svg
              className="w-3 h-3 mr-1"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
            </svg>
            {companyAnalysisBookmarkCount}
          </span>
          {bookmark && (
            <span className="ml-2 text-yellow-500">
              <svg
                className="w-3 h-3"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default CorporateReportCard;
