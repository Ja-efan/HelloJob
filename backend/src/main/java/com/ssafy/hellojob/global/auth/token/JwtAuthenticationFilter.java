package com.ssafy.hellojob.global.auth.token;

import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.util.JwtUtil;
import com.ssafy.hellojob.global.util.SseUtil;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Collections;

import static com.ssafy.hellojob.global.exception.ErrorCode.USER_NOT_FOUND;

@Component
@RequiredArgsConstructor
@Slf4j
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final UserRepository userRepository;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {

        if (shouldNotFilter(request)) {
            filterChain.doFilter(request, response); //다음 필터로 제어를 넘김.
            return;
        }

        String token = jwtUtil.extractAccessToken(request);
        if (token == null) {
            if (response.isCommitted()) return;
            if (SseUtil.isSseRequest(request)) {
                SseUtil.writeSseError(response, HttpServletResponse.SC_UNAUTHORIZED, "Unauthorized\n\n");
                response.getWriter().flush();
            } else {
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            }
        }

        try {
            authenticateUserWithToken(token, request);
        } catch (Exception e) {
            log.error("사용자 인증 설정 실패: {}", e.getMessage());
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            return;
        }

        filterChain.doFilter(request, response);
    }


    private void authenticateUserWithToken(String token, HttpServletRequest request) throws BaseException {
        String email = jwtUtil.getEmailFromToken(token);
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new BaseException(USER_NOT_FOUND));
        // 인증 정보 설정
        UserPrincipal userPrincipal = createUserPrincipal(user);
        setAuthentication(userPrincipal, request);
    }

    private UserPrincipal createUserPrincipal(User user) {
        return new UserPrincipal(
                Collections.singletonList(new SimpleGrantedAuthority("ROLE_USER")),
                user
        );
    }

    /*Spring Security는 SecurityContextHolder를 통해 현재 요청을 보낸 사용자가 누구인지 확인합니다.
    하지만 기본적으로 Spring Security는 세션 기반 인증을 사용하기 때문에,
    JWT 인증을 사용할 때는 인증된 사용자를 직접 SecurityContext에 등록해야 합니다.

    즉, 이 메서드가 없으면 JWT 검증은 성공하더라도, Spring Security가 사용자를 인식하지 못합니다.
    그러면 인증이 필요한 API 요청에서 403 Forbidden 오류가 발생할 수 있습니다.*/
    private void setAuthentication(UserPrincipal userPrincipal, HttpServletRequest request) {
        UsernamePasswordAuthenticationToken authentication =
                new UsernamePasswordAuthenticationToken(
                        userPrincipal,
                        null,
                        userPrincipal.getAuthorities()
                );

        authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
        SecurityContextHolder.getContext().setAuthentication(authentication);
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getServletPath();
        return path.startsWith("/oauth2") ||
                path.startsWith("/login") ||
                path.equals("/api/v1/auth/refresh");
    }
}
