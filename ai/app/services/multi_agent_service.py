from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
# MCP 어댑터는 호환성 문제로 직접 구현
from langsmith import Client, traceable
import operator
import os
import uuid

from app.core.mcp_core import get_mcp_servers
from app.core.logger import app_logger

logger = app_logger

# LangSmith 설정
def setup_langsmith():
    """LangSmith 추적 설정"""
    # 환경 변수에서 LangSmith 설정 확인
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if not langsmith_api_key:
        logger.warning("⚠️ LANGSMITH_API_KEY가 설정되지 않았습니다. LangSmith 추적이 비활성화됩니다.")
        return None
    
    # LangSmith 추적 활성화
    os.environ["LANGSMITH_TRACING"] = "true"
    
    # LangSmith 클라이언트 초기화
    client = Client(api_key=langsmith_api_key)
    
    logger.info("✅ LangSmith 추적이 활성화되었습니다.")
    return client

# LangSmith 클라이언트 초기화
langsmith_client = setup_langsmith()

# MCP 설정을 LangGraph 스타일로 변환
async def setup_mcp_tools():
    """기존 MCP 서버를 LangGraph에서 사용할 수 있도록 변환"""
    try:
        # 기존 get_mcp_servers 함수 사용
        mcp_servers = get_mcp_servers()
        
        if not mcp_servers:
            logger.warning("⚠️ MCP 서버가 설정되지 않았습니다.")
            return [], None
        
        # MCP 서버를 LangChain Tool 형식으로 변환
        tools = []
        tool_names = set()  # 중복 도구 이름 추적
        
        # 각 MCP 서버에서 사용 가능한 도구들을 LangChain Tool로 변환
        for server_instance in mcp_servers:
            try:
                # 서버에서 도구 목록 가져오기
                if hasattr(server_instance, 'list_tools'):
                    server_tools = await server_instance.list_tools()
                    
                    # 디버깅: server_tools의 구조 확인
                    # print(f"🔍 디버깅: {server_instance.name} 서버의 tools 구조: {type(server_tools)}")
                    # print(f"🔍 디버깅: tools 내용: {server_tools}")
                    
                    # server_tools가 리스트인지 객체인지 확인하여 처리
                    if hasattr(server_tools, 'tools'):
                        # 객체 형태인 경우
                        tools_list = server_tools.tools
                    elif isinstance(server_tools, list):
                        # 리스트 형태인 경우
                        tools_list = server_tools
                    else:
                        # 기타 경우 (None이거나 다른 타입)
                        logger.warning(f"⚠️ {server_instance.name} 서버의 tools 형태를 인식할 수 없습니다: {type(server_tools)}")
                        continue
                    
                    for tool_info in tools_list:
                        from langchain_core.tools import BaseTool
                        
                        # 🔧 도구 이름 충돌 방지: 서버 이름을 prefix로 추가
                        original_tool_name = tool_info.name
                        unique_tool_name = f"{server_instance.name}_{original_tool_name}"
                        
                        # 중복 체크 및 로깅
                        if original_tool_name in tool_names:
                            logger.warning(f"⚠️ 도구 이름 충돌 감지: '{original_tool_name}' -> '{unique_tool_name}'로 변경")
                        else:
                            tool_names.add(original_tool_name)
                        
                        # 도구 정보 로깅
                        logger.info(f"📋 로드 중인 도구: {unique_tool_name} (원본: {original_tool_name}, 서버: {server_instance.name})")
                        
                        # 🔧 클로저 문제 해결: 팩토리 함수로 변수 고정
                        def create_mcp_tool(server_inst, orig_tool_name, unique_name, tool_desc):
                            """MCP 도구 팩토리 함수 - 클로저 변수 고정"""
                            
                            class MCPTool(BaseTool):
                                name: str = unique_name
                                description: str = tool_desc
                                args_schema: type = None  # 유연한 인수 허용
                                
                                def _run(self, query: str = "", **kwargs) -> str:
                                    """도구 실행"""
                                    import asyncio
                                    return asyncio.run(self._arun(**kwargs))
                                
                                async def _arun(self, **kwargs) -> str:
                                    """비동기 도구 실행"""
                                    try:
                                        # 전달된 인수들을 정리하여 MCP 도구에 전달
                                        tool_args = {}
                                        
                                        # 추가 키워드 인수들을 tool_args에 병합
                                        if kwargs:
                                            logger.info(f"🔧 도구 '{unique_name}'({orig_tool_name})에 추가 인수 전달: {kwargs}")
                                            tool_args.update(kwargs)
                                        
                                        logger.info(f"🛠️ MCP 도구 '{unique_name}' 호출 (원본: {orig_tool_name}, 서버: {server_inst.name}) - 인수: {tool_args}")
                                        
                                        # MCP 서버의 도구 호출 (원본 도구 이름 사용)
                                        result = await server_inst.call_tool(
                                            orig_tool_name,  # 고정된 원본 이름으로 호출
                                            tool_args
                                        )
                                        return str(result.content[0].text if result.content else "실행 완료")
                                    except Exception as e:
                                        logger.error(f"❌ MCP 도구 '{unique_name}' 실행 오류: {e}")
                                        return f"도구 실행 오류: {str(e)}"
                            
                            return MCPTool()
                        
                        # 팩토리 함수로 도구 생성
                        mcp_tool = create_mcp_tool(
                            server_instance, 
                            original_tool_name, 
                            unique_tool_name,
                            f"[{server_instance.name}] {tool_info.description or f'{original_tool_name} tool from {server_instance.name}'}"
                        )
                        tools.append(mcp_tool)
                        
            except Exception as e:
                logger.error(f"⚠️ {server_instance.name} 서버 도구 로드 실패: {e}")
                continue
        
        logger.info(f"✅ MCP에서 {len(tools)}개의 도구를 로드했습니다: {[tool.name for tool in tools]}")
        return tools, mcp_servers
            
    except Exception as e:
        logger.error(f"❌ MCP 도구 설정 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # MCP 도구가 없어도 기본 동작은 가능하도록 빈 리스트 반환
        return [], None

def get_langsmith_config_guide():
    """LangSmith 설정 가이드 반환"""
    guide = """
    📊 LangSmith 추적 설정 가이드
    
    1. 환경 변수 설정:
       export LANGSMITH_API_KEY="your-langsmith-api-key"
       export LANGSMITH_TRACING="true"
    
    2. 선택적 설정:
       export LANGSMITH_PROJECT="company_analysis_multi_agent"
       export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    
    3. LangSmith 웹 UI에서 추적 결과 확인:
       - 프로젝트: company_analysis_multi_agent
       - 각 에이전트 노드별 실행 시간 및 결과
       - 전체 워크플로우 시각화
       - 에러 및 성능 분석
    
    ✅ 현재 LangSmith 상태: {"활성화" if langsmith_client else "비활성화"}
    """
    return guide

def validate_langsmith_setup():
    """LangSmith 설정 검증"""
    checks = {
        "API_KEY": os.getenv("LANGSMITH_API_KEY") is not None,
        "TRACING": os.getenv("LANGSMITH_TRACING") == "true",
        "CLIENT": langsmith_client is not None
    }
    
    print("🔍 LangSmith 설정 검증:")
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}: {status}")
    
    if all(checks.values()):
        print("🎉 LangSmith 설정이 완료되었습니다!")
        return True
    else:
        print("⚠️ LangSmith 설정을 확인해주세요.")
        print(get_langsmith_config_guide())
        return False


# 기업 분석 멀티 AI 에이전트 

# 기업 분석 결과 모델 정의
class CompanySwotData(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)
    strength_tags: List[str] = Field(default_factory=list)
    weakness_tags: List[str] = Field(default_factory=list)
    opportunity_tags: List[str] = Field(default_factory=list)
    threat_tags: List[str] = Field(default_factory=list)
    swot_summary: str = ""

class CompanyBasicInformation(BaseModel):
    company_name: str = ""
    company_brand: str = ""
    company_vision: str = ""
    company_finance_summary: str = ""
    company_news_summary: str = ""
    company_swot: CompanySwotData = Field(default_factory=CompanySwotData)
    company_analysis_summary: str = ""

class CompanyAnalysisMultiAgentOutput(BaseModel):
    company_basic_information: CompanyBasicInformation = Field(default_factory=CompanyBasicInformation)

# LangGraph 상태 정의
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    company_name: str
    user_prompt: str
    next: str
    sender: str
    # 각 에이전트의 결과를 저장
    basic_info_result: str
    dart_result: str  
    news_result: str
    swot_result: str
    final_analysis: CompanyAnalysisMultiAgentOutput
    
    
# LangGraph 에이전트 노드 함수들
@traceable(
    name="basic_info_agent_node",
    project_name="company_analysis_multi_agent",
    tags=["기업분석", "기본정보", "멀티에이전트"]
)
async def basic_info_agent_node(state: AgentState):
    """기업 기본 정보 수집 에이전트"""
    tools, mcp_client = await setup_mcp_tools()
    
    try:
        # LangGraph의 create_react_agent 사용
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        
        agent = create_react_agent(
            llm,
            tools,
            prompt=f"""당신은 '제트'라는 이름의 기업 분석 전문 AI 에이전트로, {state['company_name']}의 최신 기업 정보를 검색하여 수집하고 분석합니다. 
반환할 기본 정보는 '주요 제품 및 브랜드(서비스)'와 '기업 비전(핵심가치)' 입니다. 
두 정보에 대한 내용을 찾을 때 까지 적절한 도구를 활용하여 검색하세요.
결과는 간결하고 구체적으로 작성해주세요."""
        )
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": f"{state['company_name']} 기업의 기본 정보를 수집해주세요."}]
        })
        
        # 결과에서 마지막 AI 메시지 추출
        final_output = result["messages"][-1].content if result["messages"] else "기본 정보 수집 실패"
        
        return {
            "messages": [f"기본정보 에이전트 완료: {state['company_name']} 기업 정보 수집"],
            "basic_info_result": final_output,
            "sender": "basic_info_agent"
        }
        
    except Exception as e:
        print(f"기본정보 에이전트 오류: {e}")
        return {
            "messages": [f"기본정보 에이전트 오류: {str(e)}"],
            "basic_info_result": f"기본 정보 수집 실패: {str(e)}",
            "sender": "basic_info_agent"
        }
    finally:
        # MCP 클라이언트 정리는 메인 함수에서 처리
        pass

@traceable(
    name="dart_agent_node", 
    project_name="company_analysis_multi_agent",
    tags=["기업분석", "공시정보", "DART", "멀티에이전트"]
)
async def dart_agent_node(state: AgentState):
    """공시 정보 분석 에이전트"""
    all_tools, mcp_client = await setup_mcp_tools()
    
    try:
        # 🎯 DART 전용 도구만 필터링
        dart_tools = [tool for tool in all_tools if tool.name.startswith('dart-mcp_')]
        
        if not dart_tools:
            print("⚠️ DART MCP 도구를 찾을 수 없습니다.")
            return {
                "messages": [f"공시정보 에이전트 오류: DART 도구 없음"],
                "dart_result": "DART 도구를 찾을 수 없어서 공시 정보 분석을 수행할 수 없습니다.",
                "sender": "dart_agent"
            }
        
        print(f"🎯 DART 에이전트에서 사용할 도구: {[tool.name for tool in dart_tools]}")
        
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        
        agent = create_react_agent(
            llm,
            dart_tools,
            prompt=f"""당신은 '제트'라는 이름의 기업 분석 전문 AI 에이전트로, {state['company_name']}의 공시 정보(DART)를 활용하여 기업 재무 분석을 진행합니다.

🔧 사용 가능한 도구:
- dart-mcp_search_disclosure: 공시 정보 검색 (전자공시시스템 DART)
- dart-mcp_search_detailed_financial_data: 상세 재무 데이터 검색
- dart-mcp_search_business_information: 사업 정보 검색
- dart-mcp_get_current_date: 현재 날짜 조회

**분석 전략:**
1. 먼저 현재 날짜를 확인하세요
2. 최근 1-2년간의 주요 공시 정보를 검색하세요
3. 재무 데이터와 사업 정보를 조합하여 분석하세요

주요 재무 지표와 사업 현황을 중심으로 분석해주세요."""
        )
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": f"{state['company_name']} 기업의 공시 정보를 분석해주세요."}]
        })
        
        final_output = result["messages"][-1].content if result["messages"] else "공시 정보 분석 실패"
        
        return {
            "messages": [f"공시정보 에이전트 완료: {state['company_name']} DART 분석"],
            "dart_result": final_output,
            "sender": "dart_agent"
        }
        
    except Exception as e:
        print(f"공시정보 에이전트 오류: {e}")
        return {
            "messages": [f"공시정보 에이전트 오류: {str(e)}"],
            "dart_result": f"공시 정보 분석 실패: {str(e)}",
            "sender": "dart_agent"
        }
    finally:
        pass

@traceable(
    name="news_agent_node",
    project_name="company_analysis_multi_agent", 
    tags=["기업분석", "뉴스분석", "멀티에이전트"]
)
async def news_agent_node(state: AgentState):
    """뉴스 분석 에이전트"""
    tools, mcp_client = await setup_mcp_tools()
    
    try:
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        
        agent = create_react_agent(
            llm,
            tools,
            prompt=f"""당신은 '제트'라는 이름의 기업 분석 전문 AI 에이전트로, {state['company_name']}와 관련된 최신 뉴스를 분석하고 요약합니다.
최근 6개월 내의 주요 뉴스를 중심으로 기업의 동향과 이슈를 파악해주세요."""
        )
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": f"{state['company_name']} 기업의 최신 뉴스를 분석해주세요."}]
        })
        
        final_output = result["messages"][-1].content if result["messages"] else "뉴스 분석 실패"
        
        return {
            "messages": [f"뉴스 에이전트 완료: {state['company_name']} 뉴스 분석"],
            "news_result": final_output,
            "sender": "news_agent"
        }
        
    except Exception as e:
        print(f"뉴스 에이전트 오류: {e}")
        return {
            "messages": [f"뉴스 에이전트 오류: {str(e)}"],
            "news_result": f"뉴스 분석 실패: {str(e)}",
            "sender": "news_agent"
        }
    finally:
        pass

@traceable(
    name="swot_agent_node",
    project_name="company_analysis_multi_agent",
    tags=["기업분석", "SWOT분석", "멀티에이전트"]
)
async def swot_agent_node(state: AgentState):
    """SWOT 분석 에이전트"""
    tools, mcp_client = await setup_mcp_tools()
    
    try:
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        
        agent = create_react_agent(
            llm,
            tools,
            prompt=f"""당신은 '제트'라는 이름의 기업 분석 전문 AI 에이전트로, {state['company_name']}의 강점(Strengths), 약점(Weaknesses), 기회(Opportunities), 위협(Threats)을 분석합니다.
각 요소별로 구체적인 사례와 근거를 제시해주세요."""
        )
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": f"{state['company_name']} 기업의 SWOT 분석을 해주세요."}]
        })
        
        final_output = result["messages"][-1].content if result["messages"] else "SWOT 분석 실패"
        
        return {
            "messages": [f"SWOT 에이전트 완료: {state['company_name']} SWOT 분석"],
            "swot_result": final_output,
            "sender": "swot_agent"
        }
        
    except Exception as e:
        print(f"SWOT 에이전트 오류: {e}")
        return {
            "messages": [f"SWOT 에이전트 오류: {str(e)}"],
            "swot_result": f"SWOT 분석 실패: {str(e)}",
            "sender": "swot_agent"
        }
    finally:
        pass

@traceable(
    name="supervisor_agent_node",
    project_name="company_analysis_multi_agent",
    tags=["기업분석", "슈퍼바이저", "멀티에이전트", "라우팅"]
)
async def supervisor_agent_node(state: AgentState):
    """슈퍼바이저 에이전트 - 다음 에이전트 결정"""
    messages = state.get("messages", [])
    
    # 각 에이전트의 완료 상태 확인
    basic_completed = state.get("basic_info_result") is not None
    dart_completed = state.get("dart_result") is not None  
    news_completed = state.get("news_result") is not None
    swot_completed = state.get("swot_result") is not None
    
    # 모든 에이전트가 완료되었으면 통합 분석으로 진행
    if basic_completed and dart_completed and news_completed and swot_completed:
        return {"next": "통합분석", "messages": ["모든 에이전트 완료, 통합 분석 시작"]}
    
    # 아직 완료되지 않은 에이전트 중 다음 실행할 에이전트 결정
    if not basic_completed:
        return {"next": "기본정보", "messages": ["기본 정보 수집 시작"]}
    elif not dart_completed:
        return {"next": "공시정보", "messages": ["공시 정보 분석 시작"]}
    elif not news_completed:
        return {"next": "뉴스분석", "messages": ["뉴스 분석 시작"]}
    elif not swot_completed:
        return {"next": "SWOT분석", "messages": ["SWOT 분석 시작"]}
    
    return {"next": "통합분석", "messages": ["모든 분석 완료, 통합 분석 진행"]}

@traceable(
    name="integration_agent_node",
    project_name="company_analysis_multi_agent",
    tags=["기업분석", "통합분석", "멀티에이전트", "최종결과"]
)
async def integration_agent_node(state: AgentState):
    """통합 분석 에이전트 - 최종 결과 생성"""
    tools, mcp_client = await setup_mcp_tools()
    
    try:
        # 모든 에이전트 결과를 통합하여 최종 분석 수행
        integration_prompt = f"""
{state['company_name']} 기업에 대한 종합 분석을 수행합니다.

== 수집된 정보 ==
기본 정보: {state.get('basic_info_result', '정보 없음')}
공시 정보: {state.get('dart_result', '정보 없음')}  
뉴스 분석: {state.get('news_result', '정보 없음')}
SWOT 분석: {state.get('swot_result', '정보 없음')}

사용자 요청: {state['user_prompt']}

위 정보들을 종합하여 CompanyAnalysisMultiAgentOutput 형식으로 최종 분석 결과를 생성해주세요.
모든 정보를 종합하여 구조화된 데이터로 정리해주세요.
"""
        
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        
        agent = create_react_agent(
            llm,
            tools,
            prompt="""당신은 기업 분석 통합 전문 AI 에이전트입니다. 
각 전문 에이전트들이 수집한 정보를 종합하여 최종적인 기업 분석 결과를 생성합니다.
결과는 구조화된 형태로 정리해주세요.""",
            response_format=CompanyAnalysisMultiAgentOutput,
        )
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": integration_prompt}]
        })
        
        # 결과를 CompanyAnalysisMultiAgentOutput 형식으로 변환
        final_output_content = result["messages"][-1].content if result["messages"] else ""
        
        # 간단한 파싱을 통해 구조화된 데이터 생성
        try:
            # 기본 구조 생성
            analysis_output = final_output_content
            
        except Exception as e:
            print(f"분석 결과 구조화 중 오류: {e}")
            analysis_output = CompanyAnalysisMultiAgentOutput()
            analysis_output.company_basic_information.company_name = state['company_name']
            analysis_output.company_basic_information.company_analysis_summary = final_output_content
        
        return {
            "messages": [f"{state['company_name']} 기업 분석 완료"],
            "final_analysis": analysis_output,
            "sender": "integration_agent"
        }
        
    except Exception as e:
        print(f"통합분석 에이전트 오류: {e}")
        analysis_output = CompanyAnalysisMultiAgentOutput()
        analysis_output.company_basic_information.company_name = state['company_name']
        analysis_output.company_basic_information.company_analysis_summary = f"분석 중 오류 발생: {str(e)}"
        
        return {
            "messages": [f"통합분석 에이전트 오류: {str(e)}"],
            "final_analysis": analysis_output,
            "sender": "integration_agent"
        }
    finally:
        pass

# 조건부 엣지 함수
def decide_next_agent(state: AgentState):
    """다음 실행할 에이전트 결정"""
    return state.get("next", "기본정보")

@traceable(
    name="company_analysis_multi_agent",
    project_name="company_analysis_multi_agent", 
    tags=["기업분석", "멀티에이전트", "전체워크플로우", "LangGraph"]
)
async def company_analysis_multi_agent(company_name, base, plus, fin, swot, user_prompt):
    """
    LangGraph를 활용한 멀티 AI 에이전트 기업 분석 서비스
    LangSmith로 전체 워크플로우 추적 및 모니터링
    """
    
    # LangSmith 설정 검증
    validate_langsmith_setup()
    
    # 워크플로우 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("슈퍼바이저", supervisor_agent_node)
    workflow.add_node("기본정보", basic_info_agent_node)
    workflow.add_node("공시정보", dart_agent_node)
    workflow.add_node("뉴스분석", news_agent_node)
    workflow.add_node("SWOT분석", swot_agent_node)
    workflow.add_node("통합분석", integration_agent_node)
    
    # 엣지 설정
    workflow.add_edge(START, "슈퍼바이저")
    
    # 슈퍼바이저에서 각 에이전트로의 조건부 엣지
    workflow.add_conditional_edges(
        "슈퍼바이저",
        decide_next_agent,
        {
            "기본정보": "기본정보",
            "공시정보": "공시정보", 
            "뉴스분석": "뉴스분석",
            "SWOT분석": "SWOT분석",
            "통합분석": "통합분석"
        }
    )
    
    # 각 에이전트 완료 후 슈퍼바이저로 복귀
    workflow.add_edge("기본정보", "슈퍼바이저")
    workflow.add_edge("공시정보", "슈퍼바이저") 
    workflow.add_edge("뉴스분석", "슈퍼바이저")
    workflow.add_edge("SWOT분석", "슈퍼바이저")
    
    # 통합분석 완료 후 종료
    workflow.add_edge("통합분석", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    # 초기 상태 설정
    initial_state = {
        "messages": [f"{company_name} 기업 분석 시작"],
        "company_name": company_name,
        "user_prompt": user_prompt,
        "next": "기본정보",
        "sender": "user",
        "basic_info_result": None,
        "dart_result": None,
        "news_result": None, 
        "swot_result": None,
        "final_analysis": None
    }
    
    # 워크플로우 실행 및 추적
    run_id = str(uuid.uuid4())
    
    # LangSmith 메타데이터 설정
    metadata = {
        "company_name": company_name,
        "user_prompt": user_prompt,
        "workflow_version": "v2.0_langgraph",
        "run_id": run_id,
        "requested_analysis": {
            "base": base,
            "plus": plus, 
            "fin": fin,
            "swot": swot
        }
    }
    
    # LangSmith에 메타데이터 로그
    if langsmith_client:
        try:
            langsmith_client.create_run(
                name=f"company_analysis_{company_name}_{run_id[:8]}",
                run_type="chain",
                inputs={"company_name": company_name, "user_prompt": user_prompt},
                extra=metadata
            )
        except Exception as e:
            print(f"⚠️ LangSmith 로깅 오류: {e}")
    
    final_state = None
    node_count = 0
    
    print(f"🚀 {company_name} 기업 분석 시작 (Run ID: {run_id[:8]})")
    
    async for output in app.astream(initial_state):
        for key, value in output.items():
            node_count += 1
            print(f"✅ 노드 '{key}' 실행 완료 ({node_count}번째 단계)")
            
            # LangSmith에 중간 결과 로깅
            if langsmith_client:
                try:
                    langsmith_client.create_run(
                        name=f"{key}_step",
                        run_type="llm", 
                        inputs={"state": str(value)[:500]},  # 너무 길면 자르기
                        outputs={"node": key, "status": "completed"},
                        extra={"step_number": node_count, "run_id": run_id}
                    )
                except Exception as e:
                    print(f"⚠️ 노드 '{key}' LangSmith 로깅 오류: {e}")
            
            if "final_analysis" in value and value["final_analysis"]:
                final_state = value
    
    print(f"🎯 분석 완료! 총 {node_count}개 단계 실행됨")
    
    # 최종 결과 반환
    if final_state and final_state.get("final_analysis"):
        result = final_state["final_analysis"]
        
        # LangSmith에 최종 결과 로깅
        if langsmith_client:
            try:
                langsmith_client.update_run(
                    run_id,
                    outputs={"final_analysis": str(result)[:1000]},
                    end_time=None
                )
            except Exception as e:
                print(f"⚠️ 최종 결과 LangSmith 로깅 오류: {e}")
        
        return result
    else:
        print("⚠️ 분석 실패: 최종 결과를 생성하지 못했습니다.")
        return CompanyAnalysisMultiAgentOutput()
    
# LangSmith 테스트 및 설정 확인 함수들
async def test_langsmith_connection():
    """LangSmith 연결 테스트"""
    if not langsmith_client:
        print("❌ LangSmith 클라이언트가 초기화되지 않았습니다.")
        return False
    
    try:
        # 간단한 테스트 런 생성
        test_run = langsmith_client.create_run(
            name="langsmith_connection_test",
            run_type="chain",
            inputs={"test": "connection"},
            outputs={"status": "success"}
        )
        print(f"✅ LangSmith 연결 성공! 테스트 Run ID: {test_run.id}")
        return True
    except Exception as e:
        print(f"❌ LangSmith 연결 실패: {e}")
        return False

def get_langsmith_dashboard_url(company_name: str = None):
    """LangSmith 대시보드 URL 생성"""
    base_url = "https://smith.langchain.com"
    project_name = "company_analysis_multi_agent"
    
    if company_name:
        return f"{base_url}/projects/{project_name}?filter=company_name:{company_name}"
    else:
        return f"{base_url}/projects/{project_name}"