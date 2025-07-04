import { Input } from "@/components/ui/input";
import GradientCard from "../../components/GradientCard";
import { useState, useEffect } from "react";
import SelectModal from "./components/SelectModal";
import { corporateListApi } from "@/api/corporateReport";
import { useQuery } from "@tanstack/react-query";
import { timeParser } from "@/hooks/timeParser";

// 인터페이스 수정
interface CorporateData {
  id: string;
  corName: string;
  corSize: string;
  industryName: string;
  region: string;
  updatedAt: string;
}

function CorporateSearch() {
  const [isModal, setIsModal] = useState(false);
  const [selectedCorporate, setSelectedCorporate] = useState("");
  const [selectedCorporateId, setSelectedCorporateId] = useState("");
  const [searchKeyword, setSearchKeyword] = useState("");
  const [debouncedKeyword, setDebouncedKeyword] = useState("");

  // 검색어 디바운스 처리
  useEffect(() => {
    const timerId = setTimeout(() => {
      setDebouncedKeyword(searchKeyword);
      //console.log(searchKeyword);
    }, 300); // 0.3초 지연

    return () => {
      clearTimeout(timerId);
    };
  }, [searchKeyword]);

  // tanstack query를 사용한 데이터 불러오기 (디바운스된 검색어 사용)
  const { data: corporateList, isLoading } = useQuery({
    queryKey: ["corporateList", debouncedKeyword],
    queryFn: async () => {
      const response = await corporateListApi.getCorporateList(
        debouncedKeyword
      );
      return response.data;
    },
  });

  // API 응답을 CorporateData 형식으로 변환
  const corporates: CorporateData[] =
    corporateList?.map((company) => ({
      id: company.id.toString(),
      corName: company.companyName,
      corSize: company.companySize,
      industryName: company.companyIndustry,
      region: company.companyLocation,
      updatedAt: company.updatedAt, // 하드코딩된 업데이트 시간
    })) || [];

  const handleCardClick = (corName: string, id: string) => {
    setSelectedCorporate(corName);
    setSelectedCorporateId(id);
    setIsModal(true);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchKeyword(e.target.value);
  };

  return (
    <div className="flex flex-col items-center justify-between h-full">
      <SelectModal
        isOpen={isModal}
        onClose={() => setIsModal(false)}
        corporateName={selectedCorporate}
        corporateId={selectedCorporateId}
      />
      <div className="flex flex-col items-center h-full">
        <div className="flex flex-col items-center justify-end w-full h-1/3 mt-[8vh] mb-[4vh]">
          <h1 className="text-5xl font-bold mb-8">분석할 기업을 검색하세요</h1>
          <Input
            className="bg-white border border-[#bdc6d8] rounded-md w-140 h-10 text-base"
            value={searchKeyword}
            onChange={handleSearchChange}
          />
        </div>
        <h2 className="w-full text-2xl font-bold mb-[2vh]">최근 분석된 기업</h2>
        <div className="flex justify-start gap-4 w-[968px] mx-auto flex-wrap pb-[212px]">
          {isLoading ? (
            <div>로딩 중...</div>
          ) : corporates.length > 0 ? (
            corporates.map((corporate) => (
              <GradientCard
                key={corporate.id}
                id={corporate.id}
                width={230}
                height={376}
                initialWidth={230}
                initialHeight={180}
                corName={corporate.corName}
                corSize={corporate.corSize}
                industryName={corporate.industryName}
                region={corporate.region}
                updatedAt={timeParser(corporate.updatedAt)}
                isGradient={true}
                onClick={() => handleCardClick(corporate.corName, corporate.id)}
              />
            ))
          ) : (
            <div>
              현재는 Dart에서 지원하는 상장 기업만 등장합니다. 검색 결과가
              없습니다.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CorporateSearch;
