interface MypageHeaderProps {
  title: string;
}

function MypageHeader({ title }: MypageHeaderProps) {
  return (
    <header className="mb-6 flex justify-center md:justify-start">
      <h1 className="text-2xl md:text-3xl font-bold">{title}</h1>
    </header>
  );
}

export default MypageHeader;
