import { Link, Outlet, useLocation, useNavigate } from "react-router";
import { useState, useEffect } from "react";
import { FaBars, FaChevronLeft } from "react-icons/fa";
import { userAuthApi } from "@/api/userAuth";
import { useAuthStore } from "@/store/userStore";

function Mypage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();
  const { setIsLoggedIn, setUserName } = useAuthStore();

  useEffect(() => {
    const checkScreenSize = () => {
      if (window.innerWidth < 768) {
        setIsSidebarOpen(false);
      } else {
        setIsSidebarOpen(true);
      }
    };

    // 초기 실행
    checkScreenSize();

    // 화면 크기 변경 시 실행
    window.addEventListener("resize", checkScreenSize);
    return () => window.removeEventListener("resize", checkScreenSize);
  }, []);

  // 네비게이션 아이템 정의
  const navItems = [
    { path: "schedule", label: "일정 관리" },
    { path: "cover-letter-list", label: "자기소개서 목록" },
    { path: "bookmarks/companies", label: "기업 분석 즐겨찾기" },
    { path: "bookmarks/jobs", label: "직무 분석 즐겨찾기" },
    { path: "interviews-videos", label: "모의 면접 영상" },
    { path: "account", label: "계정 설정" },
  ];

  // 로그아웃 처리 함수
  const handleLogout = async () => {
    try {
      // 토큰이 만료되어 403 에러가 발생해도 로그아웃 처리를 진행
      await userAuthApi.logout();
      console.log("로그아웃 성공");
    } catch (error) {
      console.log("로그아웃 API 호출 실패, 로컬 로그아웃 진행", error);
    } finally {
      // 로그아웃 API 성공/실패 상관없이 항상 실행
      setIsLoggedIn(false);
      setUserName("");
      navigate("/");
    }
  };

  return (
    <div className="flex bg-[#F8F9FC] min-h-screen relative">
      {/* 모바일 메뉴 토글 버튼 */}
      <button
        className={`md:hidden fixed z-20 top-4 left-4 p-2 rounded-md border-2 transition-all duration-300 ease-in-out ${
          isSidebarOpen ? "bg-white left-57" : "bg-white text-black"
        }`}
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? <FaChevronLeft size={20} /> : <FaBars size={20} />}
      </button>

      {/* 사이드바 */}
      <div
        className={`
          ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"} 
          md:translate-x-0
          transition-transform duration-300 ease-in-out
          fixed z-10
          w-56 h-full bg-white flex flex-col items-center justify-between 
          p-5 border-r border-[#E4E8F0] shadow-md
        `}
      >
        <header className="w-full">
          <Link className="w-full flex justify-center" to="/">
            <div className="font-bold text-3xl">
              <span>HELLO</span>
              <span className="text-primary">JOB</span>
            </div>
          </Link>
        </header>
        {/* 네비게이션 메뉴 */}
        <nav className="w-full mt-6">
          <ul className="space-y-2">
            {navItems.map((item) => (
              <li key={item.path}>
                <Link
                  to={`/mypage/${item.path}`}
                  className={`block px-4 py-2 rounded-md transition-colors ${
                    location.pathname === "/mypage" &&
                    item.label === "일정 관리"
                      ? "bg-primary text-white"
                      : location.pathname === `/mypage/${item.path}`
                      ? "bg-primary text-white"
                      : "hover:bg-gray-100"
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <footer className="w-full mt-auto">
          <div className="w-full flex justify-center">
            <button
              onClick={handleLogout}
              className="px-4 py-2 rounded-md transition-colors text-center cursor-pointer"
            >
              로그아웃
            </button>
          </div>
        </footer>
      </div>

      {/* 메인 컨텐츠 영역 */}
      <Outlet />
    </div>
  );
}

export default Mypage;
