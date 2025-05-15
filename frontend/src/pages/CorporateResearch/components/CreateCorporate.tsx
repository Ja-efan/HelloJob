import { Button } from "@/components/Button";

interface CreateCorporateProps {
  onClose: () => void;
}

function CreateCorporate({ onClose }: CreateCorporateProps) {
  return (
    <div className="h-[90vh] w-[940px] bg-white rounded-t-xl py-8 px-12 overflow-y-auto">
      <Button onClick={onClose}>닫기</Button>
    </div>
  );
}

export default CreateCorporate;
