import Header from "@/components/Common/Header";
import WaveAnimation from "@/components/WaveAnimation";
import { FaRegBuilding } from "react-icons/fa";
import { HiOutlineDocumentText } from "react-icons/hi";
import { PiNotePencilBold } from "react-icons/pi";
import { MdPeopleAlt } from "react-icons/md";

import LocateBtn from "./components/LocateBtn";

export default function Home() {
  return (
    <div className="relative flex flex-col w-full h-screen overflow-hidden">
      <Header isMinimize={true} />
      <WaveAnimation />
      {/* 기존 홈페이지 내용 */}
      <main className="grow max-w-screen-xl mx-auto z-10 w-full px-10 pb-20 flex flex-col justify-center">
        <div>
          <h1 className="text-9xl font-bold mb-2">
            HELLO<span className="text-[#6F52E0]">JOB</span>
          </h1>
          <h2 className="text-4xl font-semibold">취업 준비 A부터 Z까지</h2>
        </div>
        <div className="flex justify-between mt-[10vh]">
          <LocateBtn
            to="/corporate-search"
            iconComponent={<HiOutlineDocumentText className="w-10 h-10" />}
            title="인적사항 입력"
            description="인적 사항을 이곳에 저장해두고 자동 입력 기능을 사용하세요!"
          />
          <LocateBtn
            to="/corporate-search"
            iconComponent={<FaRegBuilding className="w-9 h-9" />}
            title="기업/직무분석"
            description="공시와 뉴스를 활용한 기업분석과 자세한 직무 분석을 확인해보세요!"
          />
          <LocateBtn
            to="/corporate-search"
            iconComponent={<PiNotePencilBold className="w-10 h-10" />}
            title="자기소개서 작성"
            description="AI를 활용해 자기소개서 초안을 작성하거나 첨삭을 받아보세요!"
          />
          <LocateBtn
            to="/corporate-search"
            iconComponent={<MdPeopleAlt className="w-10 h-10" />}
            title="모의 면접"
            description="예상 질문을 확인하고 모의 면접을 통해 자신감을 키워보세요!"
          />
        </div>
      </main>
    </div>
  );
}
