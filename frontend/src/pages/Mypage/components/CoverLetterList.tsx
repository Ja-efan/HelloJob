import { Button } from "@/components/Button";
import MypageHeader from "./MypageHeader";
import { useState } from "react";
import { FaPlus, FaSearch } from "react-icons/fa";
import { Link } from "react-router";

function CoverLetterList() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("최신순");
  const [currentPage, setCurrentPage] = useState(1);
  const [coverLetters] = useState([
    {
      coverLetterId: 1,
      title: "삼성전자 하드웨어 엔지니어 자원서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "컴퓨터공학 전공자로서 반도체 분야 및 메모리 시스템 개발에서 3년간 연구 경험이 있습니다...",
      updatedAt: "2023년 4월 7일",
    },
    {
      coverLetterId: 2,
      title: "카카오 백엔드 개발자 - 경력 기술서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "고성능 서비스 개발 및 대규모 트래픽 처리 경험으로, Spring과 Kotlin을 활용하여...",
      updatedAt: "2023년 3월 23일",
    },
    {
      coverLetterId: 3,
      title: "라인 모바일 앱 개발자 자기소개서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "Flutter와 Dart를 기반으로 크로스플랫폼 앱 개발 경험이 풍부합니다. UI/UX에 대한 이해도가 높으며...",
      updatedAt: "2023년 2월 15일",
    },
    {
      coverLetterId: 4,
      title: "라인 모바일 앱 개발자 자기소개서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "Flutter와 Dart를 기반으로 크로스플랫폼 앱 개발 경험이 풍부합니다. UI/UX에 대한 이해도가 높으며...",
      updatedAt: "2023년 2월 15일",
    },
    {
      coverLetterId: 5,
      title: "라인 모바일 앱 개발자 자기소개서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "Flutter와 Dart를 기반으로 크로스플랫폼 앱 개발 경험이 풍부합니다. UI/UX에 대한 이해도가 높으며...",
      updatedAt: "2023년 2월 15일",
    },
    {
      coverLetterId: 6,
      title: "라인 모바일 앱 개발자 자기소개서",
      companyName: "기업",
      jobRoleName: "IT/웹",
      description:
        "Flutter와 Dart를 기반으로 크로스플랫폼 앱 개발 경험이 풍부합니다. UI/UX에 대한 이해도가 높으며...",
      updatedAt: "2023년 2월 15일",
    },
  ]);

  // 페이지네이션 설정
  const itemsPerPage = 5;
  const totalPages = Math.ceil(coverLetters.length / itemsPerPage);

  // 검색어에 따른 필터링
  const filteredCoverLetters = coverLetters.filter(
    (letter) =>
      letter.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      letter.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 현재 페이지에 표시할 항목들
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filteredCoverLetters.slice(
    indexOfFirstItem,
    indexOfLastItem
  );

  // 페이지 변경 핸들러
  const handlePageChange = (pageNumber: number) => {
    setCurrentPage(pageNumber);
  };

  // 선택 카테고리 변경 핸들러
  const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCategory(e.target.value);
  };

  // 검색어 변경 핸들러
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1); // 검색 시 첫 페이지로 이동
  };

  return (
    <div className="w-full p-4 md:p-6 md:ml-56 transition-all duration-300">
      <MypageHeader title="자기소개서 목록" />

      <div className="flex justify-between items-center mb-6">
        {/* 검색 UI */}
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="w-full md:w-auto">
            <select
              className="px-4 py-2 border rounded-md bg-white w-full md:w-auto focus:outline-none focus:ring-1 focus:ring-primary"
              value={selectedCategory}
              onChange={handleCategoryChange}
            >
              <option value="최신순">최신순</option>
              {/* <option value="오래된순">오래된순</option> */}
              {/* <option value="기업명">기업명</option>
              <option value="직무명">직무명</option> */}
            </select>
          </div>
          <div className="relative w-full md:w-64">
            <input
              type="text"
              placeholder="검색어를 입력하세요..."
              className="w-full px-4 py-2 pr-10 border rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-primary"
              value={searchTerm}
              onChange={handleSearchChange}
            />
            <button className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
              <FaSearch />
            </button>
          </div>
        </div>
        <Button variant="default">
          <Link to="/mypage/cover-letter/new" className="flex items-center">
            <FaPlus className="mr-2" /> 자기소개서 작성
          </Link>
        </Button>
      </div>

      <div className="space-y-4">
        {currentItems.length > 0 ? (
          currentItems.map((coverLetter) => (
            <div
              key={coverLetter.coverLetterId}
              className="bg-white cursor-pointer p-6 rounded-lg shadow-sm border border-gray-100 pl-[27px] hover:pl-6 hover:shadow-md hover:border-l-primary hover:border-l-4 hover:rounded-l-sm transition-shadow"
            >
              <div className="flex justify-between items-start mb-3">
                <h3 className="text-lg font-medium">{coverLetter.title}</h3>
                <div className="flex space-x-2">
                  <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">
                    {coverLetter.companyName}
                  </span>
                  <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">
                    {coverLetter.jobRoleName}
                  </span>
                </div>
              </div>
              <p className="text-gray-600 mb-3 text-sm line-clamp-2">
                {coverLetter.description}
              </p>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-500">
                  작성일: {coverLetter.updatedAt}
                </span>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-10 text-gray-500">
            검색 결과가 없습니다
          </div>
        )}
      </div>

      {/* 페이지네이션 (하단) */}
      {filteredCoverLetters.length > itemsPerPage && (
        <div className="flex justify-center mt-6">
          <nav className="flex space-x-1">
            <button
              onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className={`px-3 py-1 rounded-md ${
                currentPage === 1
                  ? "text-gray-400 cursor-not-allowed"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              이전
            </button>
            {Array.from({ length: totalPages }, (_, i) => i + 1).map(
              (number) => (
                <button
                  key={number}
                  onClick={() => handlePageChange(number)}
                  className={`px-3 py-1 rounded-md cursor-pointer ${
                    currentPage === number
                      ? "bg-primary text-white"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  {number}
                </button>
              )
            )}
            <button
              onClick={() =>
                handlePageChange(Math.min(totalPages, currentPage + 1))
              }
              disabled={currentPage === totalPages}
              className={`px-3 py-1 rounded-md ${
                currentPage === totalPages
                  ? "text-gray-400 cursor-not-allowed"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              다음
            </button>
          </nav>
        </div>
      )}
    </div>
  );
}

export default CoverLetterList;
