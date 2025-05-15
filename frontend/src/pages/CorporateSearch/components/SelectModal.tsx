import { useEffect, useRef } from "react";
import { Link } from "react-router";
import { FaRegChartBar } from "react-icons/fa";
import { SlPeople } from "react-icons/sl";

interface SelectModalProps {
  isOpen: boolean;
  onClose: () => void;
  corporateName: string;
  corporateId: string;
}

function SelectModal({
  isOpen,
  onClose,
  corporateName,
  corporateId,
}: SelectModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  // ESC 키를 누를 때 모달 닫기
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // 모달 외부 클릭 시 닫기
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={modalRef}
        className="bg-white rounded-lg shadow-lg w-[36rem] max-w-[90vw] max-h-[80vh] overflow-auto"
      >
        <div className="flex flex-col p-6">
          <h2 className="text-2xl font-bold text-center mb-6">
            분석 유형 선택
          </h2>

          <p className="text-lg text-center mb-6">
            {corporateName} 기업에 대해 어떤 분석을 진행하시겠습니까?
          </p>

          <div className="flex gap-6 justify-center mb-4">
            <Link
              to={`/corporate-research/${corporateId}`}
              className="w-40 h-40 flex flex-col items-center justify-center hover:bg-[#6F52E0]/10 rounded-lg transition-colors border-2 border-[#6F52E0] p-4"
            >
              <div className="w-16 h-16 bg-[#6F52E0]/30 rounded-full flex items-center justify-center mb-2">
                <FaRegChartBar className="text-[#6F52E0] w-6 h-6" />
              </div>
              <span className="font-bold text-lg text-[#6F52E0]">
                기업 분석
              </span>
            </Link>

            <Link
              to={`/job-research/${corporateId}`}
              className="w-40 h-40 flex flex-col items-center justify-center hover:bg-[#6F52E0]/10 rounded-lg transition-colors border-2 border-[#6F52E0] p-4"
            >
              <div className="w-16 h-16 bg-[#6F52E0]/30 rounded-full flex items-center justify-center mb-2">
                <SlPeople className="text-[#6F52E0] w-6 h-6" />
              </div>
              <span className="font-bold text-lg text-[#6F52E0]">
                직무 분석
              </span>
            </Link>
          </div>

          <button
            onClick={onClose}
            className="cursor-pointer py-2 px-4 border border-[#6F52E0] text-[#6F52E0] rounded-md hover:bg-[#6F52E0]/10 transition-colors self-center"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}

export default SelectModal;
