package com.ssafy.hellojob.domain.interview.service;

import com.ssafy.hellojob.domain.interview.entity.CoverLetterInterview;
import com.ssafy.hellojob.domain.interview.entity.Interview;
import com.ssafy.hellojob.domain.interview.entity.InterviewAnswer;
import com.ssafy.hellojob.domain.interview.entity.InterviewVideo;
import com.ssafy.hellojob.domain.interview.repository.InterviewAnswerRepository;
import com.ssafy.hellojob.domain.user.service.UserReadService;
import com.ssafy.hellojob.global.exception.BaseException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.ssafy.hellojob.global.exception.ErrorCode.GET_VIDEO_LENGTH_FAIL;
import static com.ssafy.hellojob.global.exception.ErrorCode.INVALID_USER;

@Slf4j
@Service
@RequiredArgsConstructor
public class InterviewAnswerSaveService {

    private final InterviewAnswerRepository interviewAnswerRepository;
    private final UserReadService userReadService;
    private final InterviewReadService interviewReadService;


    @Value("${FFPROBE_PATH}")
    private String ffprobePath;

    @Value("${FFMPEG_PATH}")
    private String ffmpegPath;

    // 한 문항 종료(면접 답변 저장)
    @Transactional
    public Map<String, String> saveInterviewAnswer(Integer userId, String url, String answer, Integer interviewAnswerId, File tempVideoFile) {

        log.debug("😎 면접 답변 저장 함수 들어옴 : {}", interviewAnswerId);

        userReadService.findUserByIdOrElseThrow(userId);

        InterviewAnswer interviewAnswer = interviewReadService.findInterviewAnswerByIdOrElseThrow(interviewAnswerId);
        InterviewVideo interviewVideo = interviewReadService.findInterviewVideoByIdOrElseThrow(interviewAnswer.getInterviewVideo().getInterviewVideoId());

        log.debug("interviewAnswerId: {}", interviewAnswer.getInterviewAnswerId());
        log.debug("interviewVideoId: {}", interviewVideo.getInterviewVideoId());

        if (interviewAnswer.getInterviewQuestionCategory().name().equals("자기소개서면접")) {
            CoverLetterInterview coverLetterInterview = interviewReadService.findCoverLetterInterviewById(interviewVideo.getCoverLetterInterview().getCoverLetterInterviewId());
            log.debug("자소서 invalid");
            log.debug("userId: {}", userId);
            log.debug("coverLetterInterviewUserId: {}", coverLetterInterview.getUser().getUserId());
            if (!userId.equals(coverLetterInterview.getUser().getUserId())) {
                throw new BaseException(INVALID_USER);
            }
        } else {
            Interview interview = interviewReadService.findInterviewById(interviewVideo.getInterview().getInterviewId());
            log.debug("면접 invalid");
            log.debug("interviewId: {}", interview.getInterviewId());
            log.debug("userId: {}", userId);
            log.debug("interviewUserId: {}", interview.getUser().getUserId());
            if (!userId.equals(interview.getUser().getUserId())) {
                throw new BaseException(INVALID_USER);
            }
        }

        String videoLength = "";
        try {
            videoLength = getVideoDurationWithFFprobe(tempVideoFile);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // interrupt 상태 복원
            log.debug("영상 길이 추출 실패 - interrupt: {}", e);
            throw new BaseException(GET_VIDEO_LENGTH_FAIL);
        } catch (IOException e) {
            log.debug("영상 길이 추출 실패 - IOException: {}", e);
            throw new BaseException(GET_VIDEO_LENGTH_FAIL);
        } catch (Exception e){
            log.debug("영상 길이 추출 실패 - Exception: {}", e);
            throw new BaseException(GET_VIDEO_LENGTH_FAIL);
        }

        interviewAnswer.addInterviewAnswer(answer);
        interviewAnswer.addInterviewVideoUrl(url);
        interviewAnswer.addVideoLength(videoLength);

        interviewAnswerRepository.flush();

        log.debug("🧪 저장 직전 answer: {}", answer);
        log.debug("🧪 저장 인터뷰 답변 ID: {}, 값: {}", interviewAnswer.getInterviewAnswerId(), interviewAnswer.getInterviewAnswer());

        return Map.of("message", "정상적으로 저장되었습니다.");
    }

    // 동영상에서 시간 뽑아내기
    // 영상 길이 추출 + .webm -> .mp4 자동 변환
    public String getVideoDurationWithFFprobe(File videoFile) throws IOException, InterruptedException {

        log.debug("😎 동영상 시간 추출 함수 들어옴");

        long start = System.nanoTime();

        // 확장자 추출
        String originalFilename = videoFile.getName();
        String extension = originalFilename.contains(".")
                ? originalFilename.substring(originalFilename.lastIndexOf("."))
                : ".webm";

        // 임시 파일 생성 및 복사
        File webmTempFile = File.createTempFile("upload", extension);
        Files.copy(videoFile.toPath(), webmTempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

        File mp4TempFile = File.createTempFile("converted", ".mp4");

        // ffmpeg 실행 (webm → mp4)
        ProcessBuilder ffmpegPb = new ProcessBuilder(
                ffmpegPath, "-y",
                "-i", webmTempFile.getAbsolutePath(),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-strict", "experimental",
                mp4TempFile.getAbsolutePath()
        );
        ffmpegPb.redirectErrorStream(true);
        Process ffmpegProcess = ffmpegPb.start();

        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(ffmpegProcess.getInputStream()))) {
                while (reader.readLine() != null); // 로그 무시
            } catch (IOException e) {
                log.warn("⚠️ ffmpeg 로그 읽기 실패", e);
            }
        }).start();

        boolean ffmpegFinished = ffmpegProcess.waitFor(30, TimeUnit.SECONDS);
        if (!ffmpegFinished) {
            ffmpegProcess.destroyForcibly();
            log.error("❌ ffmpeg 시간 초과로 강제 종료됨");
            throw new IOException("ffmpeg 변환 시간 초과");
        }

        // ffprobe 실행
        ProcessBuilder ffprobePb = new ProcessBuilder(
                ffprobePath,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                mp4TempFile.getAbsolutePath()
        );
        Process ffprobeProcess = ffprobePb.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(ffprobeProcess.getInputStream()));
        String durationStr = reader.readLine();
        ffprobeProcess.waitFor();

        try {
            Files.deleteIfExists(webmTempFile.toPath());
            Files.deleteIfExists(mp4TempFile.toPath());
        } catch (IOException e) {
            log.warn("⚠️ 임시 파일 삭제 실패", e);
        }

        if (durationStr == null || durationStr.trim().isEmpty() || durationStr.trim().equalsIgnoreCase("N/A")) {
            log.warn("⚠️ ffprobe 결과로부터 duration 추출 실패: '{}'", durationStr);
            return "";
        }

        double durationInSeconds;
        try {
            durationInSeconds = Double.parseDouble(durationStr.trim());
        } catch (NumberFormatException e) {
            log.error("❌ duration 값이 유효하지 않음: '{}'", durationStr);
            return "";
        }

        int hours = (int) durationInSeconds / 3600;
        int minutes = ((int) durationInSeconds % 3600) / 60;
        int seconds = (int) durationInSeconds % 60;

        String result = String.format("%02d:%02d:%02d", hours, minutes, seconds);
        long end = System.nanoTime();
        log.info("🎥 영상 길이: {} (처리 시간: {} ms)", result, (end - start) / 1_000_000);
        return result;
    }

}
