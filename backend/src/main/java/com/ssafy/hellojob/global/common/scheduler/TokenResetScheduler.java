package com.ssafy.hellojob.global.common.scheduler;

import com.ssafy.hellojob.domain.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class TokenResetScheduler {

    private final UserRepository userRepository;

    // 매일 자정에 실행
    @Scheduled(cron = "0 0 0 * * ?")
    public void resetTokens() {
        userRepository.resetAllTokens(3); // 모든 사용자의 토큰을 3으로 리셋
    }
}