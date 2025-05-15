import { Button } from "@/components/Button";
import Loading from "@/components/Loading/Loading";
import { useGetCompanies } from "@/hooks/companyHooks";
import { useSelectCompanyStore } from "@/store/coverLetterAnalysisStore";
import { CompanyState } from "@/types/coverLetterStoreTypes";
import { useEffect, useState } from "react";
import { AiOutlineSearch } from "react-icons/ai";

interface CompanySearchProps {
  modalClose: () => void;
}

function CompanySearch({ modalClose }: CompanySearchProps) {
  const [inputValue, setInputValue] = useState("");
  const [keyword, setKeyword] = useState("");
  const { setSelectCompany } = useSelectCompanyStore();
  const { data, isLoading } = useGetCompanies(keyword);

  useEffect(() => {
    const timer = setTimeout(() => {
      setKeyword(inputValue);
      console.log(`${keyword} 검색 실행`);
    }, 300);

    return () => {
      clearTimeout(timer);
    };
  }, [inputValue]);

  const handleOnChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleOnClick = (companyData: CompanyState) => {
    setSelectCompany(companyData);
    modalClose();

    console.log(companyData);
  };
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      modalClose();
    }
  };

  const contents = {
    title: "기업명을 검색해 주세요",
    placeholder: "기업명 입력",
  };
  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-container w-100 h-auto max-h-70">
        <div className="font-bold text-xl mb-2">{contents.title}</div>
        <div className="flex">
          <input
            type="text"
            placeholder={contents.placeholder}
            className="relative border-2 h-10 pl-3 w-full rounded-l-xl outline-0 focus:border-primary focus:border-2 duration-100"
            onChange={handleOnChange}
          />

          <Button className="w-10 h-10 rounded-l-none rounded-r-xl">
            <AiOutlineSearch className="size-5"></AiOutlineSearch>
          </Button>
        </div>
        {isLoading && <Loading radius={3}></Loading>}
        {data && (
          <div className="border border-b-0 rounded overflow-y-auto mt-3">
            {data.map((el) => (
              <div
                onClick={() =>
                  handleOnClick({
                    companyId: el.id,
                    companyLocation: el.companyLocation,
                    companyName: el.companyName,
                    companySize: el.companySize,
                  })
                }
                key={el.id}
                className="cursor-pointer text-sm border-b-1 grid grid-cols-10 items-center h-11 px-3 hover:border-primary hover:border-1 hover:rounded transition-all duration-150 hover:bg-secondary-light group:active:bg-primary  "
              >
                <span className=" col-span-1 group-active:bg-active group-active:border-active duration-100 border bg-secondary-light rounded-full size-7 flex justify-center items-center font-semibold text-primary text-sm aspect-square">
                  {el.companyName[0]}
                </span>
                <span className="col-span-3 font-medium text-left ml-3">
                  {el.companyName}
                </span>
                <span className="col-span-3 text-right ">
                  {el.companyLocation}
                </span>
                <span className="col-span-3 text-right ">{el.companySize}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default CompanySearch;
