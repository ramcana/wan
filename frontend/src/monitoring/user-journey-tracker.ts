interface UserJourneyEvent {
  id: string;
  type: 'page_view' | 'user_action' | 'api_call' | 'error' | 'performance';
  timestamp: number;
  sessionId: string;
  userId?: string;
  page: string;
  action?: string;
  data: Record<string, any>;
  duration?: number;
}

interface UserSession {
  id: string;
  userId?: string;
  startTime: number;
  endTime?: number;
  events: UserJourneyEvent[];
  metadata: {
    userAgent: string;
    viewport: { width: number; height: number };
    referrer: string;
    utm: Record<string, string>;
  };
}

interface JourneyFunnel {
  name: string;
  steps: string[];
  conversionRates: number[];
  dropOffPoints: Array<{ step: string; rate: number }>;
}

class UserJourneyTracker {
  private currentSession: UserSession;
  private events: UserJourneyEvent[] = [];
  private funnels: Map<string, JourneyFunnel> = new Map();
  private pageStartTime: number = Date.now();

  constructor() {
    this.currentSession = this.createSession();
    this.setupEventListeners();
    this.trackPageView();
    this.defineFunnels();
  }

  private createSession(): UserSession {
    const urlParams = new URLSearchParams(window.location.search);
    const utm = {
      source: urlParams.get('utm_source') || '',
      medium: urlParams.get('utm_medium') || '',
      campaign: urlParams.get('utm_campaign') || '',
      term: urlParams.get('utm_term') || '',
      content: urlParams.get('utm_content') || '',
    };

    return {
      id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      startTime: Date.now(),
      events: [],
      metadata: {
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        referrer: document.referrer,
        utm,
      },
    };
  }

  private setupEventListeners() {
    // Track page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.trackEvent('user_action', 'page_hidden');
      } else {
        this.trackEvent('user_action', 'page_visible');
      }
    });

    // Track clicks
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;
      const tagName = target.tagName.toLowerCase();
      const id = target.id;
      const className = target.className;
      const text = target.textContent?.slice(0, 50) || '';

      this.trackEvent('user_action', 'click', {
        element: tagName,
        id,
        className,
        text,
        x: event.clientX,
        y: event.clientY,
      });
    });

    // Track form submissions
    document.addEventListener('submit', (event) => {
      const form = event.target as HTMLFormElement;
      const formData = new FormData(form);
      const data: Record<string, any> = {};
      
      formData.forEach((value, key) => {
        // Don't track sensitive data
        if (!key.toLowerCase().includes('password') && 
            !key.toLowerCase().includes('token')) {
          data[key] = value;
        }
      });

      this.trackEvent('user_action', 'form_submit', {
        formId: form.id,
        formAction: form.action,
        fieldCount: formData.entries.length,
        data,
      });
    });

    // Track scroll depth
    let maxScrollDepth = 0;
    const trackScrollDepth = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      const scrollDepth = Math.round((scrollTop + windowHeight) / documentHeight * 100);

      if (scrollDepth > maxScrollDepth) {
        maxScrollDepth = scrollDepth;
        
        // Track milestone scroll depths
        if (scrollDepth >= 25 && maxScrollDepth < 25) {
          this.trackEvent('user_action', 'scroll_25');
        } else if (scrollDepth >= 50 && maxScrollDepth < 50) {
          this.trackEvent('user_action', 'scroll_50');
        } else if (scrollDepth >= 75 && maxScrollDepth < 75) {
          this.trackEvent('user_action', 'scroll_75');
        } else if (scrollDepth >= 100 && maxScrollDepth < 100) {
          this.trackEvent('user_action', 'scroll_100');
        }
      }
    };

    let scrollTimeout: NodeJS.Timeout;
    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(trackScrollDepth, 100);
    });

    // Track time on page
    window.addEventListener('beforeunload', () => {
      const timeOnPage = Date.now() - this.pageStartTime;
      this.trackEvent('user_action', 'page_exit', { timeOnPage });
      this.endSession();
    });
  }

  private defineFunnels() {
    // Generation funnel
    this.funnels.set('generation', {
      name: 'Video Generation',
      steps: [
        'page_view_generation',
        'form_start',
        'form_submit',
        'queue_view',
        'generation_complete',
      ],
      conversionRates: [],
      dropOffPoints: [],
    });

    // Onboarding funnel
    this.funnels.set('onboarding', {
      name: 'User Onboarding',
      steps: [
        'page_view_home',
        'click_get_started',
        'page_view_generation',
        'first_generation_submit',
        'first_generation_complete',
      ],
      conversionRates: [],
      dropOffPoints: [],
    });

    // Gallery engagement funnel
    this.funnels.set('gallery', {
      name: 'Gallery Engagement',
      steps: [
        'page_view_gallery',
        'video_click',
        'video_play',
        'video_share',
      ],
      conversionRates: [],
      dropOffPoints: [],
    });
  }

  trackEvent(
    type: UserJourneyEvent['type'],
    action?: string,
    data: Record<string, any> = {},
    duration?: number
  ) {
    const event: UserJourneyEvent = {
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      timestamp: Date.now(),
      sessionId: this.currentSession.id,
      userId: this.currentSession.userId,
      page: window.location.pathname,
      action,
      data,
      duration,
    };

    this.events.push(event);
    this.currentSession.events.push(event);

    // Update funnel progress
    this.updateFunnelProgress(event);

    // Send to analytics
    this.sendToAnalytics(event);

    // Log in development
    if (process.env.NODE_ENV === 'development') {
      console.log('📊 User Journey Event:', event);
    }
  }

  trackPageView(page?: string) {
    const currentPage = page || window.location.pathname;
    this.pageStartTime = Date.now();

    this.trackEvent('page_view', `page_view${currentPage.replace(/\//g, '_')}`, {
      url: window.location.href,
      title: document.title,
      referrer: document.referrer,
    });
  }

  trackUserAction(action: string, data?: Record<string, any>) {
    this.trackEvent('user_action', action, data);
  }

  trackApiCall(endpoint: string, method: string, duration: number, success: boolean) {
    this.trackEvent('api_call', `${method}_${endpoint}`, {
      endpoint,
      method,
      success,
    }, duration);
  }

  trackError(error: string, context?: Record<string, any>) {
    this.trackEvent('error', error, context);
  }

  trackPerformance(metric: string, value: number, context?: Record<string, any>) {
    this.trackEvent('performance', metric, { value, ...context });
  }

  private updateFunnelProgress(event: UserJourneyEvent) {
    this.funnels.forEach((funnel, funnelName) => {
      const eventKey = event.action || `${event.type}_${event.page}`;
      const stepIndex = funnel.steps.indexOf(eventKey);
      
      if (stepIndex !== -1) {
        // User reached this step in the funnel
        this.calculateFunnelMetrics(funnelName);
      }
    });
  }

  private calculateFunnelMetrics(funnelName: string) {
    const funnel = this.funnels.get(funnelName);
    if (!funnel) return;

    const sessions = this.getSessionsWithFunnelEvents(funnel.steps);
    const conversionRates: number[] = [];
    const dropOffPoints: Array<{ step: string; rate: number }> = [];

    for (let i = 0; i < funnel.steps.length; i++) {
      const step = funnel.steps[i];
      const sessionsReachingStep = sessions.filter(session =>
        session.events.some(event => 
          (event.action || `${event.type}_${event.page}`) === step
        )
      ).length;

      const conversionRate = sessions.length > 0 ? (sessionsReachingStep / sessions.length) * 100 : 0;
      conversionRates.push(conversionRate);

      if (i > 0) {
        const previousStepSessions = sessions.filter(session =>
          session.events.some(event => 
            (event.action || `${event.type}_${event.page}`) === funnel.steps[i - 1]
          )
        ).length;

        const dropOffRate = previousStepSessions > 0 ? 
          ((previousStepSessions - sessionsReachingStep) / previousStepSessions) * 100 : 0;

        dropOffPoints.push({ step, rate: dropOffRate });
      }
    }

    funnel.conversionRates = conversionRates;
    funnel.dropOffPoints = dropOffPoints;
  }

  private getSessionsWithFunnelEvents(steps: string[]): UserSession[] {
    // This would typically query a database, but for now we'll use in-memory data
    const allSessions = [this.currentSession]; // In a real app, this would be all sessions
    
    return allSessions.filter(session =>
      steps.some(step =>
        session.events.some(event =>
          (event.action || `${event.type}_${event.page}`) === step
        )
      )
    );
  }

  private sendToAnalytics(event: UserJourneyEvent) {
    // Send to Google Analytics
    if (window.gtag) {
      window.gtag('event', event.action || event.type, {
        event_category: event.type,
        event_label: event.page,
        custom_map: event.data,
        value: event.duration,
      });
    }

    // Send to custom analytics endpoint
    if (process.env.NODE_ENV === 'production') {
      fetch('/api/v1/analytics/journey', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event),
      }).catch(console.error);
    }
  }

  setUserId(userId: string) {
    this.currentSession.userId = userId;
  }

  endSession() {
    this.currentSession.endTime = Date.now();
    
    // Send session summary
    const sessionDuration = this.currentSession.endTime - this.currentSession.startTime;
    const eventCount = this.currentSession.events.length;
    const uniquePages = new Set(this.currentSession.events.map(e => e.page)).size;

    this.sendToAnalytics({
      id: `session_end_${this.currentSession.id}`,
      type: 'user_action',
      timestamp: Date.now(),
      sessionId: this.currentSession.id,
      userId: this.currentSession.userId,
      page: window.location.pathname,
      action: 'session_end',
      data: {
        duration: sessionDuration,
        eventCount,
        uniquePages,
        metadata: this.currentSession.metadata,
      },
    });
  }

  // Analytics and reporting methods
  getSessionSummary(): {
    duration: number;
    eventCount: number;
    pageViews: number;
    userActions: number;
    apiCalls: number;
    errors: number;
  } {
    const now = Date.now();
    const duration = now - this.currentSession.startTime;
    const events = this.currentSession.events;

    return {
      duration,
      eventCount: events.length,
      pageViews: events.filter(e => e.type === 'page_view').length,
      userActions: events.filter(e => e.type === 'user_action').length,
      apiCalls: events.filter(e => e.type === 'api_call').length,
      errors: events.filter(e => e.type === 'error').length,
    };
  }

  getFunnelReport(funnelName: string): JourneyFunnel | null {
    return this.funnels.get(funnelName) || null;
  }

  getAllFunnels(): JourneyFunnel[] {
    return Array.from(this.funnels.values());
  }

  getEventsByType(type: UserJourneyEvent['type']): UserJourneyEvent[] {
    return this.events.filter(e => e.type === type);
  }

  getEventsByTimeRange(start: number, end: number): UserJourneyEvent[] {
    return this.events.filter(e => e.timestamp >= start && e.timestamp <= end);
  }

  // User behavior insights
  getUserBehaviorInsights(): {
    mostVisitedPages: Array<{ page: string; count: number }>;
    mostCommonActions: Array<{ action: string; count: number }>;
    averageSessionDuration: number;
    bounceRate: number;
  } {
    const events = this.currentSession.events;
    const pageViews = events.filter(e => e.type === 'page_view');
    const userActions = events.filter(e => e.type === 'user_action');

    // Most visited pages
    const pageCount: Record<string, number> = {};
    pageViews.forEach(e => {
      pageCount[e.page] = (pageCount[e.page] || 0) + 1;
    });
    const mostVisitedPages = Object.entries(pageCount)
      .map(([page, count]) => ({ page, count }))
      .sort((a, b) => b.count - a.count);

    // Most common actions
    const actionCount: Record<string, number> = {};
    userActions.forEach(e => {
      if (e.action) {
        actionCount[e.action] = (actionCount[e.action] || 0) + 1;
      }
    });
    const mostCommonActions = Object.entries(actionCount)
      .map(([action, count]) => ({ action, count }))
      .sort((a, b) => b.count - a.count);

    // Session metrics
    const sessionDuration = Date.now() - this.currentSession.startTime;
    const bounceRate = pageViews.length <= 1 ? 100 : 0; // Simplified bounce rate

    return {
      mostVisitedPages,
      mostCommonActions,
      averageSessionDuration: sessionDuration,
      bounceRate,
    };
  }
}

export const userJourneyTracker = new UserJourneyTracker();

// React hook for user journey tracking
export const useUserJourneyTracker = () => {
  return {
    trackEvent: userJourneyTracker.trackEvent.bind(userJourneyTracker),
    trackPageView: userJourneyTracker.trackPageView.bind(userJourneyTracker),
    trackUserAction: userJourneyTracker.trackUserAction.bind(userJourneyTracker),
    trackApiCall: userJourneyTracker.trackApiCall.bind(userJourneyTracker),
    trackError: userJourneyTracker.trackError.bind(userJourneyTracker),
    trackPerformance: userJourneyTracker.trackPerformance.bind(userJourneyTracker),
    setUserId: userJourneyTracker.setUserId.bind(userJourneyTracker),
  };
};