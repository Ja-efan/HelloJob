// const sampleData = `
// ((기본))
// (사업 개요) : {현대오토에버는 IT서비스와 차량용 소프트웨어 사업을 영위함. IT서비스는 SI(시스템 통합)와 ITO(IT 아웃소싱) 부문으로 구성, 정보시스템 기획·구축·운영·유지보수 전과정을 수행. 차량용 SW는 내비게이션 SW, 차량 SW 플랫폼 등을 개발·판매하며, 자율주행과 커넥티비티 등 차세대 미래차 기술 영역에 집중함.}
// (주요 제품/서비스) : {주요 사업은 ITO(IT시스템 운영·관리 등, 매출비중 약 44%), SI(IT컨설팅·시스템 설계·개발, 약 34%), 차량용 SW(플랫폼/내비게이션, 약 22%)로 구분됨. 차량 SW 브랜드로는 mobilgene Classic, mobilgene Adaptive 등이 있으며, 내비게이션 SW로 '지니(GINI)' 시리즈를 OEM 및 PIO방식으로 공급.}
// ((심화))
// (원재료 및 설비) : {주요 원재료는 전산장비(서버 등, 매입비중 28%)와 외주용역(매입비중 72%)으로, 소프트웨어 및 IT 인프라가 사업의 기반. 주요 설비는 파주·광주 데이터센터와 재해복구센터임. 물적 제조가 아닌 용역서비스 제공이 중심.}
// `;

export interface Section {
  type: "title" | "subtitle" | "content";
  content: string;
}

export const parseData = (data: string): Section[] => {
  // 중괄호 내부의 개행을 제거하는 전처리
  const preprocessData = (text: string): string => {
    return text.replace(/\{([^}]+)\}/g, (_, content) => {
      // 중괄호 내부의 개행과 연속된 공백을 단일 공백으로 변환
      const cleanContent = content.replace(/\s+/g, " ").trim();
      return `{${cleanContent}}`;
    });
  };

  const preprocessedData = preprocessData(data);
  const lines = preprocessedData.trim().split("\n");
  const result: Section[] = [];
  const titleRegex = /\(\((.+)\)\)/;
  const subtitleRegex = /\((.+)\) : {(.+)}/;

  for (const line of lines) {
    const titleMatch = titleRegex.exec(line);
    if (titleMatch) {
      result.push({ type: "title", content: titleMatch[1] });
      continue;
    }

    const subtitleMatch = subtitleRegex.exec(line);
    if (subtitleMatch) {
      result.push({
        type: "subtitle",
        content: subtitleMatch[1],
      });
      result.push({
        type: "content",
        content: subtitleMatch[2],
      });
    }
  }

  console.log(result);
  return result;
};
