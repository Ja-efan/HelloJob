import React, { useEffect, useRef } from "react";
import { Button } from "./Button";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm?: () => void;
  title?: string;
  children: React.ReactNode;
  width?: string;
  height?: string;
  warning?: boolean;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  children,
  width = "25rem",
  height = "auto",
  warning = false,
}) => {
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
      className="modal-overlay"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={modalRef}
        className="modal-container"
        style={{
          width,
          height,
          maxHeight: "90vh",
        }}
      >
        <div className="flex flex-col gap-2 flex-1">
          {title && (
            <h2 className="text-xl font-bold text-foreground flex items-center">
              {warning ? (
                <div className="text-red-500 bg-red-500/20 rounded-full w-7 h-7 flex items-center justify-center font-extrabold text-xl mr-2">
                  !
                </div>
              ) : (
                <div className="text-[#6F52E0] bg-[#6F52E0]/20 rounded-full w-7 h-7 flex items-center justify-center font-extrabold text-xl mr-2">
                  i
                </div>
              )}
              {title}
            </h2>
          )}
          <div className="modal-content px-2 pt-2">{children}</div>
          <footer className="flex gap-2 mt-4 justify-end">
            <Button onClick={onClose} variant={"white"}>
              <span className="px-3">취소</span>
            </Button>
            {onConfirm && (
              <Button onClick={onConfirm} variant={"default"}>
                <span className="px-3">확인</span>
              </Button>
            )}
          </footer>
        </div>
      </div>
    </div>
  );
};

export default Modal;
