package com.ssafy.hellojob.global.auth.controller;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping
@Slf4j
public class TestController {

    @GetMapping("/test")
    public String test() {
        return "test";
    }

    @GetMapping("/api/v1/test")
    public ResponseEntity<Map<String, String>> test1() {
        Map<String, String> response = new HashMap<>();
        response.put("text", "ok");
        return ResponseEntity.ok(response);
    }

    @GetMapping("/")
    public String getAccessToken(HttpServletRequest request) {
        log.info("루트 경로 핸들러 호출됨");
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            log.info("쿠키 수: {}", cookies.length);
            for (Cookie cookie : cookies) {
                log.info("쿠키 이름: {}, 값: {}", cookie.getName(), cookie.getValue());
                if (cookie.getName().equals("access_token")) {
                    return cookie.getValue();
                }
            }
        } else {
            log.info("쿠키가 없음");
        }
        return "쿠키가 없습니다.";
    }
}
