import { Button } from "@/components/Button";
import { useCoverLetterStore } from "@/store/coverLetterStore";
import { useCallback, useEffect, useRef, useState } from "react";
import { AiOutlineSend } from "react-icons/ai";

export interface InputChatProps {
  onSubmitMessage: () => void;
}

function InputChat({ onSubmitMessage }: InputChatProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [inputValue, setInputValue] = useState("");
  const { addUserMessage } = useCoverLetterStore();
  const isComposing = useRef(false);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [inputValue]);

  const handleOnChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  const handleAddMessage = (message: string) => {
    addUserMessage(message);
    onSubmitMessage();
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      if (isComposing.current) {
        return;
      }

      e.preventDefault();
      if (inputValue.trim() !== "") {
        handleAddMessage(inputValue);
      }
    }
  };

  const handleCompositionStart = useCallback(() => {
    isComposing.current = true;
  }, []);

  const handleCompositionEnd = useCallback(() => {
    isComposing.current = false;
  }, []);

  return (
    <>
      <div className="min-h-[3rem] mt-7 duration-100 border-2 border-transparent ring ring-[#b8b9ba44]  bg-background  rounded-2xl focus-within:border-accent focus-within:border-2">
        <div className="relative w-full">
          <label htmlFor="userMessage"></label>
          <textarea
            ref={textareaRef}
            id="userMessage"
            name="userMessage"
            value={inputValue}
            className="break-all resize-none px-3 py-2 pr-14 max-h-40 overflow-y-auto outline-0 w-full"
            onChange={handleOnChange}
            onKeyDown={handleKeyDown}
            onCompositionStart={handleCompositionStart}
            onCompositionEnd={handleCompositionEnd}
            rows={1}
            placeholder="메시지를 입력하세요"
          ></textarea>
          <div className="absolute right-3 bottom-1.5 ">
            <Button
              aria-label="메시지 전송"
              className="flex-shrink-0"
              onClick={() => handleAddMessage(inputValue)}
              disabled={inputValue.trim() === ""}
            >
              <AiOutlineSend className="size-4" />
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}

export default InputChat;
